"""PPO updates for variable batches, preserving complete recurrent rally sequences."""

import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from pong_rl.models import RecurrentAgent


def replay_rally(agent, rally, device, checkpoint_steps=32):
    """Compatibility wrapper for a single complete rally."""
    return replay_rallies(agent, [rally], device, checkpoint_steps)


def replay_rallies(agent, rallies, device, checkpoint_steps=32, checkpoint_threshold=0):
    """Replay padded rallies together; return only real steps in rally order.

    Each lane starts with its own saved memory. Padding follows real steps and
    contributes no loss. Chunk boundaries preserve the full recurrent gradient.
    Small batches can retain activations; larger batches recompute checkpoint
    chunks to bound accelerator memory. Neither mode truncates gradients.
    """
    lengths = [len(r) for r in rallies]
    batch = len(rallies)
    steps = max(lengths)
    observations = torch.zeros((steps, batch, *rallies[0].observations.shape[1:]),
                               dtype=rallies[0].observations.dtype)
    actions = torch.zeros((steps, batch), dtype=torch.long)
    for lane, rally in enumerate(rallies):
        observations[:len(rally), lane] = rally.observations
        actions[:len(rally), lane] = rally.actions
    state = tuple(torch.cat([r.initial_state[i] for r in rallies], dim=1).to(device)
                  for i in range(2))
    outputs = [[], [], []]

    def replay(obs, act, h, c):
        _, logp, entropy, value, (h, c) = agent.get_action_and_value(
            obs.flatten(0, 1).to(device=device, dtype=torch.float32), (h, c),
            torch.zeros(act.numel(), device=device), act.flatten().to(device))
        return logp, entropy, value.flatten(), h, c

    use_checkpoint = steps * batch > checkpoint_threshold
    for start in range(0, steps, checkpoint_steps):
        inputs = (observations[start:start+checkpoint_steps],
                  actions[start:start+checkpoint_steps], *state)
        result = (checkpoint(replay, *inputs, use_reentrant=False)
                  if use_checkpoint else replay(*inputs))
        state = result[3:]
        for chunks, result_tensor in zip(outputs, result[:3]):
            chunks.append(result_tensor.reshape(-1, batch))
    # Known CPU lengths avoid CUDA boolean indexing and its dynamic-size sync.
    indices = torch.cat([torch.arange(length) * batch + lane
                         for lane, length in enumerate(lengths)]).to(device)
    return tuple(torch.cat(chunks).flatten().index_select(0, indices) for chunks in outputs)


def update_policy(agent, optimizer, batch, args, device):
    rallies = batch.rallies
    offsets = np.cumsum([0] + [len(r) for r in rallies])
    old_logp = torch.cat([r.logprobs for r in rallies]).to(device)
    old_values = torch.cat([r.values for r in rallies]).to(device)
    advantages = torch.cat([r.advantages for r in rallies])
    returns = torch.cat([r.returns for r in rallies]).to(device)
    if args.norm_adv:
        # Normalize over the actual batch for both architectures, including tiny remainders.
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    advantages = advantages.to(device)
    recurrent = isinstance(agent, RecurrentAgent)
    observations = None if recurrent else torch.cat([r.observations for r in rallies])
    actions = torch.cat([r.actions for r in rallies])
    totals = {key: torch.zeros((), device=device) for key in ("value_loss", "policy_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac")}
    samples = 0
    optimizer_steps = 0

    for _ in range(args.update_epochs):
        epoch_kl, epoch_samples = torch.zeros((), device=device), 0
        if recurrent:
            # Balance real step counts while preserving complete rallies.
            order = np.random.permutation(len(rallies)).tolist()
            order.sort(key=lambda i: len(rallies[i]), reverse=True)
            groups = [[] for _ in range(min(args.num_minibatches, len(rallies)))]
            sizes = [0] * len(groups)
            for i in order:
                target = int(np.argmin(sizes))
                groups[target].append(i)
                sizes[target] += len(rallies[i])
            # Length balancing must not fix the optimizer's group order every epoch.
            order = np.random.permutation(len(groups))
            groups, sizes = [groups[i] for i in order], [sizes[i] for i in order]
        else:
            groups = np.array_split(np.random.permutation(batch.steps), min(args.num_minibatches, batch.steps))
            sizes = [len(group) for group in groups]

        for group, size in zip(groups, sizes):
            optimizer.zero_grad()
            if recurrent:
                indices = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in group])
                units = [(indices, group)]
            else:
                # Accumulate CNN microbatches too, to bound accelerator image/activation storage.
                units = [(group[start:start+256], None) for start in range(0, size, 256)]
            for indices, rally_index in units:
                if recurrent:
                    newlogp, entropy, newvalue = replay_rallies(
                        agent, [rallies[i] for i in rally_index], device,
                        checkpoint_threshold=args.replay_checkpoint_threshold)
                else:
                    _, newlogp, entropy, newvalue = agent.get_action_and_value(
                        observations[indices].to(device=device, dtype=torch.float32), actions[indices].to(device))
                    newvalue = newvalue.flatten()
                logratio = newlogp - old_logp[indices]
                ratio = logratio.exp()
                adv = advantages[indices]
                pg_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1-args.clip_coef, 1+args.clip_coef)).mean()
                targets = returns[indices]
                if args.clip_vloss:
                    old = old_values[indices]
                    clipped = old + (newvalue - old).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.maximum((newvalue-targets).square(), (clipped-targets).square()).mean()
                else:
                    v_loss = 0.5 * (newvalue-targets).square().mean()
                entropy_loss = entropy.mean()
                count = len(newlogp)
                # One optimizer step per minibatch: weight by transitions, not rally count.
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                (loss * (count / size)).backward()
                with torch.no_grad():
                    kl = ((ratio - 1) - logratio).mean()
                    metrics = dict(value_loss=v_loss.detach(), policy_loss=pg_loss.detach(),
                                   entropy=entropy_loss.detach(), old_approx_kl=(-logratio).mean(),
                                   approx_kl=kl, clipfrac=((ratio-1).abs() > args.clip_coef).float().mean())
                for key, value in metrics.items():
                    totals[key] += value * count
                samples += count
                epoch_kl += kl * count
                epoch_samples += count
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
        if args.target_kl is not None and epoch_kl / epoch_samples > args.target_kl:
            break

    variance = returns.var(unbiased=False).item()
    explained = 1 - (returns-old_values).var(unbiased=False).item() / variance if variance else float("nan")
    metric_values = (torch.stack(list(totals.values())) / samples).cpu().tolist()
    if not all(math.isfinite(value) for value in metric_values):
        raise FloatingPointError("Non-finite PPO loss; stopping training")
    return {**{f"losses/{key}": value for key, value in zip(totals, metric_values)},
            "losses/explained_variance": explained, "charts/optimizer_steps": optimizer_steps}

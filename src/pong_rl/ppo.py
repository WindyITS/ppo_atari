"""PPO updates; recurrent agents replay whole sequences from their saved memory."""

import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from pong_rl.models import RecurrentAgent


def replay_sequences(agent, sequences, device, checkpoint_steps=32, checkpoint_threshold=0,
                     observations=None, actions=None):
    """Replay padded sequences together; return only real steps in sequence order.

    observations and actions optionally give each sequence's images and actions already on the
    accelerator (one upload per update instead of one per minibatch and epoch); by default the
    sequences' own CPU tensors are used.

    Each lane starts with its own saved memory. Padding follows real steps and
    contributes no loss. Chunk boundaries preserve the full recurrent gradient.
    Small batches can retain activations; larger batches recompute checkpoint
    chunks to bound accelerator memory. Neither mode truncates gradients.
    """
    lengths = [len(s) for s in sequences]
    batch = len(sequences)
    steps = max(lengths)
    sources = [s.observations for s in sequences] if observations is None else observations
    action_sources = [s.actions for s in sequences] if actions is None else actions
    padded = torch.zeros((steps, batch, *sources[0].shape[1:]), dtype=sources[0].dtype,
                         device=sources[0].device)
    padded_actions = torch.zeros((steps, batch), dtype=torch.long, device=action_sources[0].device)
    for lane, (length, source, action) in enumerate(zip(lengths, sources, action_sources)):
        padded[:length, lane] = source
        padded_actions[:length, lane] = action
    observations, actions = padded, padded_actions.to(device)
    state = tuple(torch.cat([s.initial_state[i] for s in sequences], dim=1).to(device)
                  for i in range(2))
    outputs = [[], [], []]

    def replay(obs, act, h, c):
        # Sequences never cross a game reset, so replay needs no done flags.
        _, logp, entropy, value, (h, c) = agent.get_action_and_value(
            obs.flatten(0, 1).to(device=device, dtype=torch.float32), (h, c),
            None, act.flatten().to(device))
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
    sequences = batch.sequences
    offsets = np.cumsum([0] + [len(s) for s in sequences])
    old_logp = torch.cat([s.logprobs for s in sequences]).to(device)
    old_values = torch.cat([s.values for s in sequences]).to(device)
    advantages = torch.cat([s.advantages for s in sequences])
    returns = torch.cat([s.returns for s in sequences]).to(device)
    if args.norm_adv:
        # Normalize over the whole batch for every architecture.
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    advantages = advantages.to(device)
    recurrent = isinstance(agent, RecurrentAgent)
    # Upload the batch's uint8 images once; every minibatch of every epoch then reads them on the
    # accelerator instead of gathering and copying from CPU memory each time. At the defaults this is
    # 4,096 steps (about 116 MB for 4 stacked frames); it grows with num_envs x num_steps.
    observations = torch.cat([s.observations for s in sequences]).to(device)
    actions = torch.cat([s.actions for s in sequences]).to(device)
    # Per-sequence views of the uploaded tensors, for recurrent replay.
    sequence_observations = [observations[offsets[i]:offsets[i+1]] for i in range(len(sequences))]
    sequence_actions = [actions[offsets[i]:offsets[i+1]] for i in range(len(sequences))]
    totals = {key: torch.zeros((), device=device) for key in ("value_loss", "policy_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac")}
    samples = 0
    optimizer_steps = 0

    for _ in range(args.update_epochs):
        epoch_kl, epoch_samples = torch.zeros((), device=device), 0
        if recurrent:
            # Balance real step counts while keeping every sequence whole.
            order = np.random.permutation(len(sequences)).tolist()
            order.sort(key=lambda i: len(sequences[i]), reverse=True)
            groups = [[] for _ in range(min(args.num_minibatches, len(sequences)))]
            sizes = [0] * len(groups)
            for i in order:
                target = int(np.argmin(sizes))
                groups[target].append(i)
                sizes[target] += len(sequences[i])
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
            for indices, sequence_index in units:
                index = torch.as_tensor(indices, device=device)  # one small upload, reused below
                if recurrent:
                    newlogp, entropy, newvalue = replay_sequences(
                        agent, [sequences[i] for i in sequence_index], device,
                        checkpoint_threshold=args.replay_checkpoint_threshold,
                        observations=[sequence_observations[i] for i in sequence_index],
                        actions=[sequence_actions[i] for i in sequence_index])
                else:
                    _, newlogp, entropy, newvalue = agent.get_action_and_value(
                        observations.index_select(0, index).float(), actions.index_select(0, index))
                    newvalue = newvalue.flatten()
                logratio = newlogp - old_logp.index_select(0, index)
                ratio = logratio.exp()
                adv = advantages.index_select(0, index)
                pg_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1-args.clip_coef, 1+args.clip_coef)).mean()
                targets = returns.index_select(0, index)
                if args.clip_vloss:
                    old = old_values.index_select(0, index)
                    clipped = old + (newvalue - old).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.maximum((newvalue-targets).square(), (clipped-targets).square()).mean()
                else:
                    v_loss = 0.5 * (newvalue-targets).square().mean()
                entropy_loss = entropy.mean()
                count = len(newlogp)
                # One optimizer step per minibatch: weight by transitions, not sequence count.
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

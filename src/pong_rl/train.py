# Adapted from CleanRL's ppo_atari.py and ppo_atari_lstm.py:
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from pong_rl.envs import make_env
from pong_rl.models import Agent, LSTMAgent, zero_lstm_state
from pong_rl.metrics import Episode, EpisodeWindow

gym.register_envs(ale_py)

# Frames per observation: the CNN sees motion through stacking, the LSTM through its memory
FRAME_STACK = {"cnn": 4, "lstm": 1}


@dataclass
class Args:
    exp_name: str = "ppo_pong"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "pong-rl"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = True
    report_window_steps: int = 500_000

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    arch: str = "cnn"
    use_fog: bool = False
    mask_enemy: bool = False
    total_timesteps: int = 20_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Computed in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_pong")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--arch",
        choices=list(FRAME_STACK),
        default="cnn",
        help="cnn: Nature CNN on 4 stacked frames; lstm: Nature CNN + LSTM on 1 frame per step",
    )
    parser.add_argument("--use-fog", action="store_true", help="Enable FogWrapper on agent half")
    parser.add_argument("--mask-enemy", action="store_true", help="Enable EnemyMaskWrapper on enemy paddle")
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project-name", type=str, default="pong-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-window-steps", type=int, default=500_000,
                        help="aggregate environment-step window for episodes ending in (step-window, step]")
    parsed = parser.parse_args()
    if parsed.report_window_steps <= 0:
        parser.error("--report-window-steps must be positive")

    return Args(
        exp_name=parsed.exp_name,
        seed=parsed.seed,
        arch=parsed.arch,
        use_fog=parsed.use_fog,
        mask_enemy=parsed.mask_enemy,
        total_timesteps=parsed.total_timesteps,
        num_envs=parsed.num_envs,
        track=parsed.track,
        wandb_project_name=parsed.wandb_project_name,
        wandb_entity=parsed.wandb_entity,
        capture_video=parsed.capture_video,
        save_model=parsed.save_model,
        report_window_steps=parsed.report_window_steps,
    )


def main():
    # Limit CPU thread contention when running multiple parallel training jobs
    torch.set_num_threads(1)

    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.arch == "lstm":
        assert args.num_envs % args.num_minibatches == 0, "LSTM minibatches are whole envs"

    condition = f"fog_{int(args.use_fog)}_mask_{int(args.mask_enemy)}"
    tag = f"{args.arch}_{condition}"
    run_name = f"pong__{tag}__seed_{args.seed}__{int(time.time())}"
    episode_window = EpisodeWindow(args.report_window_steps)
    os.makedirs("logs", exist_ok=True)
    episode_log_path = os.path.join("logs", f"{run_name}_episodes.jsonl")
    print(f"Episode outcomes: {episode_log_path}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            group=tag,
            name=run_name,
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[{run_name}] Using device: {device}")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                use_fog=args.use_fog,
                mask_enemy=args.mask_enemy,
                frame_stack=FRAME_STACK[args.arch],
            )
            for i in range(args.num_envs)
        ],
        # The GAE computation below assumes the returned obs after a done is already the reset obs
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert envs.single_observation_space.shape is not None

    agent = LSTMAgent(envs).to(device) if args.arch == "lstm" else Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup: (128 steps, 8 envs, ...)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    # LSTM memory, carried across steps and across rollouts (unused by the CNN agent)
    next_lstm_state = zero_lstm_state(args.num_envs, device)

    os.makedirs("checkpoints", exist_ok=True)

    for iteration in range(1, args.num_iterations + 1):
        # The update replays each env's rollout through the LSTM starting from this state
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # 1. Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 2. Rollout phase: collect 128 steps across 8 environments
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if isinstance(agent, LSTMAgent):
                    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                        next_obs, next_lstm_state, next_done
                    )
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # With SAME_STEP autoreset, finished episodes are logged in infos["final_info"]["episode"]
            # with a boolean mask infos["final_info"]["_episode"]
            final_info = infos.get("final_info", {})
            if "episode" in final_info:
                for i, has_ep in enumerate(final_info["_episode"]):
                    if has_ep:
                        ep_return = float(final_info["episode"]["r"][i])
                        ep_length = int(final_info["episode"]["l"][i])
                        episode = Episode(global_step, ep_return, ep_length,
                                          bool(terminations[i]), bool(truncations[i]))
                        episode_window.add(episode)
                        with open(episode_log_path, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"arch": args.arch, "seed": args.seed,
                                                     "condition": condition, **episode.to_dict()},
                                                    allow_nan=False) + "\n")
                        print(f"[{tag}|s{args.seed}] step={global_step}, return={ep_return}, "
                              f"length={ep_length}, outcome={episode.outcome}, "
                              f"terminated={int(episode.terminated)}, truncated={int(episode.truncated)}")
                        if args.track:
                            wandb.log(
                                {
                                    "charts/episodic_return": ep_return,
                                    "charts/episodic_length": ep_length,
                                    "charts/episode_terminated": int(episode.terminated),
                                    "charts/episode_truncated": int(episode.truncated),
                                    "charts/episode_outcome": episode.outcome,
                                    "global_step": global_step,
                                }
                            )

        # 3. GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            if isinstance(agent, LSTMAgent):
                next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch: (128, 8, ...) -> (1024, ...)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 4. PPO Update phase: 4 epochs over the 1024 steps in minibatches of 256.
        # The CNN samples random steps; the LSTM must replay whole sequences in order,
        # so each of its minibatches is all 128 steps of num_envs / num_minibatches envs.
        b_inds = np.arange(args.batch_size)
        env_inds = np.arange(args.num_envs)
        envs_per_batch = args.num_envs // args.num_minibatches
        flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            if isinstance(agent, LSTMAgent):
                np.random.shuffle(env_inds)
            else:
                np.random.shuffle(b_inds)
            for mb in range(args.num_minibatches):
                if isinstance(agent, LSTMAgent):
                    mb_env_inds = env_inds[mb * envs_per_batch : (mb + 1) * envs_per_batch]
                    # Time-major (step 0 of each env, step 1 of each env, ...), as get_states expects
                    mb_inds = flat_inds[:, mb_env_inds].ravel()
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        (initial_lstm_state[0][:, mb_env_inds], initial_lstm_state[1][:, mb_env_inds]),
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                else:
                    mb_inds = b_inds[mb * args.minibatch_size : (mb + 1) * args.minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (Clipped surrogate objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        window_metrics = episode_window.metrics(global_step)

        if iteration % 25 == 0:
            print(f"[{tag}|s{args.seed}] iter={iteration}/{args.num_iterations}, step={global_step}, SPS={sps}")
            print(f"[{tag}|s{args.seed}] reporting_window="
                  f"({window_metrics['start_step_exclusive']},{global_step}], "
                  f"episodes={window_metrics['episodes']}, mean_return={window_metrics['mean_return']}, "
                  f"mean_length={window_metrics['mean_episode_length']}, "
                  f"completed_games={window_metrics['completed_games']}, "
                  f"completed_win_rate={window_metrics['completed_win_rate']}, "
                  f"timeouts={window_metrics['timeouts']}")

        if args.track:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": sps,
                    "global_step": global_step,
                    **{f"charts/window/{key}": value for key, value in window_metrics.items()},
                }
            )

    if args.save_model:
        ckpt_path = f"checkpoints/{tag}_seed_{args.seed}.pt"
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "args": vars(args),
                "arch": args.arch,
                "condition": condition,
                "seed": args.seed,
            },
            ckpt_path,
        )
        print(f"Saved model checkpoint to {ckpt_path}")
        if args.track:
            wandb.save(ckpt_path)

    if args.track:
        wandb.finish()
    envs.close()


if __name__ == "__main__":
    main()

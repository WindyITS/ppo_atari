import ale_py
import cv2
import gymnasium as gym
import numpy as np

from pong_rl.wrappers import EnemyMaskWrapper, FogWrapper

gym.register_envs(ale_py)


def get_state_frame(use_fog: bool, mask_enemy: bool) -> np.ndarray:
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )
    if use_fog:
        env = FogWrapper(env)
    if mask_enemy:
        env = EnemyMaskWrapper(env)
    obs, _ = env.reset(seed=42)
    for _ in range(53):
        obs, _, _, _, _ = env.step(0)
    return obs


def main():
    configs = [
        ("off_off", False, False, "1. Fog: OFF | Enemy Mask: OFF"),
        ("off_on", False, True, "2. Fog: OFF | Enemy Mask: ON"),
        ("on_off", True, False, "3. Fog: ON  | Enemy Mask: OFF"),
        ("on_on", True, True, "4. Fog: ON  | Enemy Mask: ON"),
    ]

    scale = 4
    panels = []
    for name, fog, enemy, title in configs:
        frame = get_state_frame(fog, enemy)
        upscaled = cv2.resize(frame, (84 * scale, 84 * scale), interpolation=cv2.INTER_NEAREST)
        color = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"assets/state_{name}.png", color)

        header = np.zeros((36, 84 * scale, 3), dtype=np.uint8)
        cv2.putText(header, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([header, color])
        panel = cv2.copyMakeBorder(panel, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        panels.append(panel)

    top_row = np.hstack([panels[0], panels[1]])
    bot_row = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top_row, bot_row])
    cv2.imwrite("assets/four_states_grid.png", grid)
    print("Saved screenshots to assets/")


if __name__ == "__main__":
    main()

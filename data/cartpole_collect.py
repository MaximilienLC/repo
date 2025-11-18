import json
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.utils.play import play


# --- 1. Custom Wrapper ---
class PausedSeededWrapper(gym.Wrapper):
    def __init__(self, env, start_seed=0):
        super().__init__(env)
        self.current_seed = start_seed
        self.is_first_reset = True
        self.last_start_time = None

    def reset(self, **kwargs):
        kwargs.pop("seed", None)

        # --- PAUSE LOGIC ---
        if not self.is_first_reset:
            print(
                f"\n>>> EPISODE FINISHED. Press [SPACE] to start Seed {self.current_seed} <<<"
            )

            waiting = True
            while waiting:
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    waiting = False
                if keys[pygame.K_ESCAPE]:
                    print("Exiting...")
                    exit()
                time.sleep(0.1)
        else:
            self.is_first_reset = False

        # --- CAPTURE TIME ---
        self.last_start_time = datetime.now().isoformat()

        # --- RESET LOGIC ---
        obs, info = self.env.reset(seed=self.current_seed, **kwargs)
        print(
            f"--> Playing Seed: {self.current_seed} (Started at {self.last_start_time})"
        )

        self.current_seed += 1
        return obs, info


# --- 2. Data Structures & RESUME LOGIC ---
output_filename = "cartpole_episodes.json"
current_episode_steps = []

# CHECK FOR EXISTING DATA
if os.path.exists(output_filename):
    print(f"Found existing data file: {output_filename}")
    with open(output_filename, "r") as f:
        try:
            all_episodes = json.load(f)
            if len(all_episodes) > 0:
                last_seed = all_episodes[-1]["seed_used"]
                start_seed = last_seed + 1
                print(
                    f"Resuming from Seed {start_seed} (Previous total episodes: {len(all_episodes)})"
                )
            else:
                print("File exists but is empty. Starting from Seed 0.")
                all_episodes = []
                start_seed = 0
        except json.JSONDecodeError:
            print("File exists but is corrupted. Starting fresh.")
            all_episodes = []
            start_seed = 0
else:
    print("No existing data found. Starting fresh from Seed 0.")
    all_episodes = []
    start_seed = 0


# --- 3. The Callback ---
def save_data_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global current_episode_steps, all_episodes

    if isinstance(obs_t, tuple):
        obs_t = obs_t[0]
    if isinstance(obs_tp1, tuple):
        obs_tp1 = obs_tp1[0]

    step_record = {
        "observation": obs_t.tolist(),
        "action": int(action),
        "reward": float(rew),
        "next_observation": obs_tp1.tolist(),
        "done": bool(terminated or truncated),
    }
    current_episode_steps.append(step_record)

    # --- END OF EPISODE LOGIC ---
    if terminated or truncated:
        episode_seed = env.current_seed - 1

        episode_data = {
            "episode_id": len(all_episodes),  # Continues the ID sequence
            "timestamp": env.last_start_time,
            "seed_used": episode_seed,
            "length": len(current_episode_steps),
            "steps": list(current_episode_steps),
        }

        all_episodes.append(episode_data)

        # --- INSTANT SAVING ---
        print(
            f"Episode {len(all_episodes)-1} finished (Seed {episode_seed}). Saving..."
        )
        with open(output_filename, "w") as f:
            json.dump(all_episodes, f, indent=2)

        current_episode_steps.clear()


# --- 4. Setup ---
base_env = gym.make("CartPole-v1", render_mode="rgb_array")
# Initialize the wrapper with the calculated start_seed
env = PausedSeededWrapper(base_env, start_seed=start_seed)

mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}

print("Controls: Left/Right Arrows.")
print("The game will PAUSE after every episode.")
print("Press SPACE to continue, or ESC to quit.")

# --- 5. Play ---
try:
    play(env, keys_to_action=mapping, callback=save_data_callback, fps=5)
except KeyboardInterrupt:
    print("\nStopped by user.")
except SystemExit:
    print("\nClosed.")

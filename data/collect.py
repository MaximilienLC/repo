import json
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.utils.play import play

# --- CONFIGURATION FOR GAMES ---
GAME_CONFIGS = {
    "1": {
        "id": "CartPole-v1",
        "name": "CartPole",
        "filename": "sub01_data_cartpole.json",
        "fps": 15,
        "noop": 0,  # If no key pressed, apply Action 0 (Push Left)
        "keys": {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1},
        "help": "Left/Right Arrows to balance.",
    },
    "2": {
        "id": "MountainCar-v0",
        "name": "MountainCar",
        "filename": "sub01_data_mountaincar.json",
        "fps": 15,
        "noop": 1,  # If no key pressed, Action 1 is "Coast" (do nothing)
        "keys": {
            (pygame.K_LEFT,): 0,  # Accelerate Left
            (pygame.K_RIGHT,): 2,  # Accelerate Right
            (pygame.K_DOWN,): 1,  # Coast explicitly
        },
        "help": "Left/Right to accelerate. Down to coast.",
    },
    "3": {
        "id": "Acrobot-v1",
        "name": "Acrobot",
        "filename": "sub01_data_acrobot.json",
        "fps": 15,
        "noop": 1,  # If no key pressed, Action 1 is "No Torque"
        "keys": {
            (pygame.K_LEFT,): 0,  # Apply -1 Torque
            (pygame.K_RIGHT,): 2,  # Apply +1 Torque
            (pygame.K_DOWN,): 1,  # Apply 0 Torque
        },
        "help": "Swing! Left/Right for torque. Down for neutral.",
    },
    "4": {
        "id": "LunarLander-v3",
        "name": "LunarLander",
        "filename": "sub01_data_lunarlander.json",
        "fps": 20,
        "noop": 0,  # If no key pressed, Action 0 is "Do Nothing"
        "keys": {
            (pygame.K_UP,): 2,  # Main Engine
            (pygame.K_LEFT,): 1,  # Left Engine
            (pygame.K_RIGHT,): 3,  # Right Engine
        },
        "help": "UP: Main Engine. LEFT/RIGHT: Side Engines.",
    },
}

# --- GLOBAL VARS TO BE SET AT RUNTIME ---
output_filename = ""
all_episodes = []
current_episode_steps = []


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


# --- 2. The Callback ---
def save_data_callback(
    obs_t, obs_tp1, action, rew, terminated, truncated, info
):
    global current_episode_steps, all_episodes, output_filename

    # Handle Tuple observations
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

        # Calculate Total Score
        total_score = sum(step["reward"] for step in current_episode_steps)

        episode_data = {
            "episode_id": len(all_episodes),
            "timestamp": env.last_start_time,
            "seed_used": episode_seed,
            "length": len(current_episode_steps),
            "score": total_score,  # Save score for easier analysis
            "steps": list(current_episode_steps),
        }

        all_episodes.append(episode_data)

        print(
            f"Episode {len(all_episodes)-1} finished (Seed {episode_seed}). Score: {total_score:.2f}. Saving to {output_filename}..."
        )

        try:
            with open(output_filename, "w") as f:
                json.dump(all_episodes, f, indent=2)
        except Exception as e:
            print(f"Error saving file: {e}")

        current_episode_steps.clear()


# --- 3. User Selection & Setup ---
print("--- GYM RECORDER ---")
print("Select a game to record:")
for key, conf in GAME_CONFIGS.items():
    print(f"[{key}] {conf['name']}")

choice = input("Enter number (default 1): ").strip()
if choice not in GAME_CONFIGS:
    choice = "1"

selected_config = GAME_CONFIGS[choice]
output_filename = selected_config["filename"]
print(f"\nSelected: {selected_config['name']}")
print(f"Output File: {output_filename}")
print(f"Controls: {selected_config['help']}")


# --- 4. Resume Logic ---
if os.path.exists(output_filename):
    print(f"Found existing data file: {output_filename}")
    with open(output_filename, "r") as f:
        try:
            all_episodes = json.load(f)
            if len(all_episodes) > 0:
                last_seed = all_episodes[-1]["seed_used"]
                start_seed = last_seed + 1
                print(
                    f"Resuming from Seed {start_seed} (Total episodes: {len(all_episodes)})"
                )
            else:
                print("File empty. Starting Seed 0.")
                start_seed = 0
        except json.JSONDecodeError:
            print("File corrupted. Starting fresh Seed 0.")
            all_episodes = []
            start_seed = 0
else:
    print("Starting fresh from Seed 0.")
    all_episodes = []
    start_seed = 0


# --- 5. Launch Game ---
try:
    base_env = gym.make(selected_config["id"], render_mode="rgb_array")
    env = PausedSeededWrapper(base_env, start_seed=start_seed)

    print("\nGame Window Opening...")
    print("Press SPACE to start the next episode when paused.")
    print("Press ESC to quit at any time.")

    play(
        env,
        keys_to_action=selected_config["keys"],
        callback=save_data_callback,
        fps=selected_config["fps"],
        noop=selected_config["noop"],  # Explicit NOOP from config
    )
except KeyboardInterrupt:
    print("\nStopped by user.")
except SystemExit:
    print("\nClosed.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback

    traceback.print_exc()  # Print full error details for debugging
    if "Box2D" in str(e):
        print(
            "TIP: For LunarLander, make sure to run: pip install 'gymnasium[box2d]'"
        )

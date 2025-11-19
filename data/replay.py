import json
import time
import os
import gymnasium as gym
import pygame  # Only needed if you want to access keys for config consistency, though not strictly needed for replay logic alone

# --- CONFIGURATION (Must match recorder) ---
GAME_CONFIGS = {
    "1": {
        "id": "CartPole-v1",
        "name": "CartPole",
        "filename": "data_cartpole.json",
        "fps": 15,
    },
    "2": {
        "id": "MountainCar-v0",
        "name": "MountainCar",
        "filename": "data_mountaincar.json",
        "fps": 15,
    },
    "3": {
        "id": "Acrobot-v1",
        "name": "Acrobot",
        "filename": "data_acrobot.json",
        "fps": 15,
    },
    "4": {
        "id": "LunarLander-v3",
        "name": "LunarLander",
        "filename": "data_lunarlander.json",
        "fps": 100,
    },
}

# --- 1. User Selection ---
print("--- GYM REPLAY ---")
print("Select a game to replay:")
for key, conf in GAME_CONFIGS.items():
    print(f"[{key}] {conf['name']}")

choice = input("Enter number (default 1): ").strip()
if choice not in GAME_CONFIGS:
    choice = "1"

selected_config = GAME_CONFIGS[choice]
filename = selected_config["filename"]
env_id = selected_config["id"]
target_fps = selected_config["fps"]

print(f"\nSelected: {selected_config['name']}")
print(f"Loading file: {filename}")

# --- 2. Load Data ---
if not os.path.exists(filename):
    print(
        f"Error: File '{filename}' not found. Have you recorded any episodes yet?"
    )
    exit()

with open(filename, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} episodes.")

# --- 3. Create Environment ---
try:
    env = gym.make(env_id, render_mode="human")
except Exception as e:
    print(f"Error creating environment: {e}")
    if "Box2D" in str(e):
        print("TIP: For LunarLander, run: pip install 'gymnasium[box2d]'")
    exit()

# --- 4. Replay Loop ---
for episode in data:
    seed = episode["seed_used"]
    steps = episode["steps"]
    episode_id = episode["episode_id"]

    # Calculate score (sum of rewards) for display
    total_score = sum(step["reward"] for step in steps)

    print(f"\n--- Replaying Episode {episode_id} ---")
    print(f"Seed: {seed} | Steps: {len(steps)} | Score: {total_score:.2f}")

    # RESET with the EXACT seed to reproduce the initial state
    obs, _ = env.reset(seed=seed)

    # Calculate sleep time based on recorded FPS
    frame_delay = 1.0 / target_fps

    for step in steps:
        action = step["action"]

        # Apply the action
        env.step(action)

        # Maintain the recorded speed
        time.sleep(frame_delay)

    # Pause after episode
    user_input = input(
        f"Episode {episode_id} finished. [ENTER] for next, [q] to quit: "
    )
    if user_input.lower() == "q":
        break

env.close()
print("Replay finished.")

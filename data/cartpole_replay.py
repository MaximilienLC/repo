import json
import time

import gymnasium as gym

# 1. Load the saved data
filename = "cartpole_episodes.json"
with open(filename, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} episodes from {filename}")

# 2. Create Environment (Render mode is Human so you can see it)
env = gym.make("CartPole-v1", render_mode="human")

# 3. Loop through every recorded episode
for episode in data:
    seed = episode["seed_used"]
    steps = episode["steps"]

    print(f"\n--- Replaying Episode {episode['episode_id']} ---")
    print(f"Seed: {seed}")
    print(f"Steps: {len(steps)}")

    # KEY STEP: Reset with the EXACT seed used during recording
    # This ensures the pole starts in the exact same position
    obs, _ = env.reset(seed=seed)

    # 4. Replay the actions exactly as recorded
    for step in steps:
        action = step["action"]

        # Apply the action
        env.step(action)

        # Wait 0.2 seconds (which equals 5 FPS)
        # If you want it faster, decrease this number (e.g., 0.05)
        time.sleep(0.01)

    # Pause at the end of the episode so you can see the result
    input("Episode finished. Press [ENTER] to watch the next one...")

env.close()

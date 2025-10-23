import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class CartPoleDataset(Dataset):
    """Dataset for CartPole trajectories from HuggingFace."""

    def __init__(self, hf_dataset):
        self.observations = []
        self.actions = []

        # Extract observations and actions from the dataset
        for episode in hf_dataset:
            obs = episode['obs']
            acts = episode['acts']

            # obs shape is (timesteps, obs_dim)
            # acts shape is (timesteps,)
            for i in range(len(acts)):
                self.observations.append(obs[i])
                self.actions.append(acts[i])

        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'observation': torch.from_numpy(self.observations[idx]),
            'action': torch.tensor(self.actions[idx], dtype=torch.long)
        }


class CartPoleDataModule(pl.LightningDataModule):
    """Lightning DataModule for CartPole behavior cloning."""

    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = None

    def prepare_data(self):
        # Download the dataset
        load_dataset("HumanCompatibleAI/ppo-CartPole-v1")

    def setup(self, stage=None):
        # Load dataset
        hf_dataset = load_dataset("HumanCompatibleAI/ppo-CartPole-v1", split='train')
        self.dataset = CartPoleDataset(hf_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False
        )


class BehaviorCloningAgent(pl.LightningModule):
    """Lightning Module for behavior cloning on CartPole."""

    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Simple linear layer
        self.policy = nn.Linear(obs_dim, action_dim)

        # Track metrics
        self.train_accuracies = []
        self.train_step_count = 0

    def forward(self, obs):
        logits = self.policy(obs)
        return logits

    def training_step(self, batch, batch_idx):
        obs = batch['observation']
        actions = batch['action']

        # Forward pass
        logits = self(obs)

        # Compute loss (cross-entropy for discrete actions)
        loss = F.cross_entropy(logits, actions)

        # Compute accuracy
        predicted_actions = torch.argmax(logits, dim=1)
        accuracy = (predicted_actions == actions).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def predict_action(self, obs):
        """Predict action for a single observation."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()

            # Move to same device as model
            obs = obs.to(self.device)

            logits = self(obs)
            action = torch.argmax(logits, dim=-1)
            return action.cpu().item()


class MetricsCallback(Callback):
    """Callback to track metrics during training."""

    def __init__(self, env: gym.Env, eval_freq: int = 100):
        self.accuracies = []
        self.steps = []
        self.current_step = 0
        self.env = env
        self.eval_freq = eval_freq
        self.rewards = []
        self.reward_steps = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get the logged accuracy
        if 'train_accuracy' in trainer.callback_metrics:
            acc = trainer.callback_metrics['train_accuracy'].item()
            self.accuracies.append(acc)
            self.steps.append(self.current_step)

            # Evaluate policy periodically
            if self.current_step % self.eval_freq == 0:
                reward = evaluate_policy(pl_module, self.env, num_episodes=3)
                self.rewards.append(reward)
                self.reward_steps.append(self.current_step)

            self.current_step += 1


def evaluate_policy(agent: BehaviorCloningAgent, env: gym.Env, num_episodes: int = 3) -> float:
    """Evaluate the agent's policy by running rollouts."""
    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.predict_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Create CartPole environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: CartPole-v1")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Create data module
    data_module = CartPoleDataModule(batch_size=64)

    # Create model
    model = BehaviorCloningAgent(obs_dim=obs_dim, action_dim=action_dim, learning_rate=1e-3)

    # Create metrics callback with environment for periodic evaluation
    metrics_callback = MetricsCallback(env=env, eval_freq=100)

    # Evaluate before training
    print("\nEvaluating policy before training...")
    reward_before = evaluate_policy(model, env, num_episodes=3)
    print(f"Average reward before training: {reward_before:.2f}")

    # Store initial reward in callback
    metrics_callback.rewards.append(reward_before)
    metrics_callback.reward_steps.append(0)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        callbacks=[metrics_callback],
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    # Train the model
    print("\nTraining behavior cloning agent...")
    trainer.fit(model, data_module)

    # Final evaluation
    print("\nEvaluating policy after training...")
    reward_after = evaluate_policy(model, env, num_episodes=3)
    print(f"Average reward after training: {reward_after:.2f}")

    # Add final reward to tracking
    metrics_callback.rewards.append(reward_after)
    metrics_callback.reward_steps.append(metrics_callback.current_step)

    # Create plots
    print("\nCreating plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot accuracy over time
    ax1.plot(metrics_callback.steps, metrics_callback.accuracies)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Prediction Accuracy Over Training')
    ax1.grid(True)

    # Plot rewards over time
    ax2.plot(metrics_callback.reward_steps, metrics_callback.rewards, marker='o')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Average Reward (3 episodes)')
    ax2.set_title('Average Reward Over Training')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('C:\\Users\\Max\\Dropbox\\repo\\code\\projects\\scaling_bc_to_perfection\\training_results.png')
    print("Plots saved to training_results.png")

    # Close environment
    env.close()

    print("\nTraining complete!")
    print(f"Final accuracy: {metrics_callback.accuracies[-1]:.4f}")
    print(f"Final average reward: {reward_after:.2f}")
    print(f"Reward improvement: {reward_before:.2f} -> {reward_after:.2f}")


if __name__ == "__main__":
    main()

# Gotchas

## Dataset Structure
- The HuggingFace dataset `HumanCompatibleAI/ppo-CartPole-v1` contains episodes with 'obs' and 'acts' keys
- Need to iterate through episodes and then through timesteps to extract individual (observation, action) pairs
- Observations are stored as lists that need to be converted to numpy arrays

## PyTorch Lightning Callbacks
- To track metrics over every training step, need to use `on_train_batch_end` callback method
- Metrics are accessible via `trainer.callback_metrics` dictionary
- For periodic evaluation during training, callbacks must have access to the environment

## Device Management
- When using the trained model for inference (e.g., during rollouts), observations must be explicitly moved to the model's device
- The model may be on GPU while environment returns CPU tensors

## Gymnasium API
- Modern Gymnasium environments return 5 values from `step()`: (obs, reward, done, truncated, info)
- Need to check both `done` and `truncated` flags to properly terminate episodes
- `reset()` returns (obs, info) tuple

## Plotting Rewards Over Time
- To plot rewards over training time, need to periodically evaluate the policy during training
- This adds computational overhead but is necessary for tracking performance progression
- Evaluation frequency (eval_freq parameter) should be balanced between plot granularity and training speed

## PyTorch Lightning Data Loading
- Setting `num_workers=0` avoids multiprocessing issues on Windows
- Setting `persistent_workers=False` when num_workers=0 prevents warnings

## Accuracy Tracking
- Using cross-entropy loss for discrete action classification
- Accuracy computed by comparing argmax of logits with ground truth actions
- Both loss and accuracy logged at every step for fine-grained tracking

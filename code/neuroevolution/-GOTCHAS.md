# Gotchas

## TorchRL Environment Wrapping

**Issue**: `ParallelEnv` from torchrl expects torchrl-wrapped environments, not raw gymnasium environments.

**Error**: `AttributeError: 'TimeLimit' object has no attribute 'fake_tensordict'`

**Solution**: Wrap gymnasium environments with `GymWrapper`:
```python
from torchrl.envs import ParallelEnv, GymWrapper

def make_env():
    return GymWrapper(gym.make('CartPole-v1'))

envs = ParallelEnv(population_size, make_env, device=device)
```

## TorchRL Action Format

**Issue**: TorchRL environments with discrete action spaces expect one-hot encoded actions, not discrete action indices.

**Error**: `RuntimeError: The size of tensor a (2) must match the size of tensor b (100) at non-singleton dimension 1`

**Details**:
- The action_spec shows `OneHot(shape=torch.Size([100, 2]), ...)`
- This means actions must be shape `[population_size, num_actions]`, not `[population_size]`

**Solution**: Convert discrete action indices to one-hot encoding:
```python
actions = population.get_actions(observations)  # [population_size]
actions_onehot = F.one_hot(actions, num_classes=2).long()  # [population_size, 2]
tensordict['action'] = actions_onehot
```

## TorchRL Environment Stepping

**Issue**: When stepping through environments in a loop, you must use the 'next' key from the tensordict to move to the next state, not call reset manually with tensors.

**Error**: `AttributeError: 'Tensor' object has no attribute 'batch_size'`

**Details**:
- After calling `envs.step(tensordict)`, the returned tensordict contains the next state under the 'next' key
- Calling `envs.reset(tensordict['next', 'done'])` passes a Tensor instead of a tensordict, causing the error
- TorchRL handles auto-resets internally when environments are done

**Solution**: Simply use the 'next' tensordict directly:
```python
for step in range(max_steps):
    # Get observations and actions...
    tensordict['action'] = actions_onehot

    # Step environments
    tensordict = envs.step(tensordict)

    # Get rewards and process them...

    # Move to next state (TorchRL handles auto-reset internally)
    tensordict = tensordict['next']
```

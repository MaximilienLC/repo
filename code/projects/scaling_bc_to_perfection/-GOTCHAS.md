# Gotchas

## Implementation Notes

1. **Data Device Management**: When using PyTorch Lightning with GPU, explicitly move the dataset tensors to GPU before passing them to the DataModule. While Lightning automatically handles model device placement, the data needs to be on the same device to avoid errors.

2. **CartPole Environment Specs**:
   - State dimension: 4 (cart position, cart velocity, pole angle, pole angular velocity)
   - Action dimension: 2 (push left or push right)
   - Action space is discrete

3. **Behavior Cloning with Constant Action**:
   - Training to imitate a policy that always outputs the same action results in trivial learning (100% accuracy very quickly)
   - This is expected since the model just needs to memorize a single output
   - Useful for verifying the training pipeline works correctly

4. **PyTorch Lightning Accuracy Tracking**:
   - Store accuracies in a list attribute of the LightningModule during training_step
   - Access this list after training to generate plots
   - Each training step corresponds to one batch

5. **Discrete Action Space Handling**:
   - Use Linear layer with output_dim = num_actions
   - Apply CrossEntropyLoss (expects logits, not probabilities)
   - Use argmax for predictions during inference
   - Action targets should be LongTensor (not Float)

6. **Episode Termination in Data Collection**:
   - Must reset environment when episode terminates or truncates
   - CartPole episodes terminate when pole angle exceeds threshold or cart moves too far
   - Always collect the requested number of transitions even if multiple episode resets are needed

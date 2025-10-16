Read code\projects\scaling_bc_to_perfection\DESCRIPTION.md. In a single file, implement the following:
1) A `gymnasium` `CartPole` environment.
2) Some logic to collect `num_transitions` state-action pairs from a random policy over that environment.
3) A PyTorch Lightning `LightningDataModule`.
4) A PyTorch Lightning `LightningModule` that represents the behaviour imitation agent. For now, simply have it make use of a PyTorch `Linear` layer with input/output dimensions corresponding to the environment's dimensions. If the output space is discrete, have one output per value and run an argmax over it. If the output space is continuous and the awaited values are bounded, have it run a `tanh` followed by a scaling factor if necessary.
5) A PyTorch Lightning `Trainer`.
6) Run the logic to collect the dataset, create a `train_dataloader` out of it and run `trainer.fit`.

Run the script and report back on accuracy over time.
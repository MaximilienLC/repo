Read `@code\projects\scaling_bc_to_perfection\---PROJECT_PROPOSAL.md` and `@code\projects\scaling_bc_to_perfection\-GOTCHAS.md`. In a single file, implement the following:
1) A `gymnasium` `CartPole` environment.
2) Some logic to collect `num_transitions` state-action pairs from a policy that always outputs the same action over that environment.
3) A PyTorch Lightning `LightningDataModule`.
4) A PyTorch Lightning `LightningModule` that represents the behaviour imitation agent. For now, simply have it make use of a PyTorch `Linear` layer with input/output dimensions corresponding to the environment's dimensions. If the output space is discrete, have one output per value and run an argmax over it. If the output space is continuous and the awaited values are bounded, have it run a `tanh` followed by a scaling factor if necessary.
5) A PyTorch Lightning `Trainer`.
6) In the main script component, have the logic to collect the dataset, create a `train_dataloader` out of it and feed it to `trainer.fit`.
Make sure to collect the accuracy over every single step and generate a plot out of it. Run only 1 epoch.

Do not read any other `.md` file. Do not investigate `amdsmi` warnings. Absolutely make sure that no relevant specification is ignored, even when running into errors. After debugging, write down any new gotchas in `@code\projects\scaling_bc_to_perfection\-GOTCHAS.md` (don't forget to take into account existing ones). Do not run any tasks in the background.
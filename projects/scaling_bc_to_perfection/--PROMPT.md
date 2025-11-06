Read `@code\projects\scaling_bc_to_perfection\-----PROJECT_PROPOSAL.md` and `@code\projects\scaling_bc_to_perfection\-GOTCHAS.md`. In a single file, implement the following:
1) A `gymnasium` `CartPole` environment.
2) A PyTorch Lightning `LightningDataModule`.
3) A PyTorch Lightning `LightningModule` that represents the behaviour imitation agent. For now, simply have it make use of a PyTorch `Linear` layer with input/output dimensions corresponding to the environment's dimensions. If the output space is discrete, have one output per value and run an argmax over it. If the output space is continuous and the awaited values are bounded, have it run a `tanh` followed by a scaling factor if necessary.
4) A PyTorch Lightning `Trainer`.
5) Pull the data from `https://huggingface.co/datasets/HumanCompatibleAI/ppo-CartPole-v1`
6) Create a `train_dataloader` out of it and feed it to `trainer.fit`.
Make sure to collect the accuracy of the prediction over every single step.
Also run some rollouts to see the average reward over 3 episodes.
Plot both of these over time.

Do not read any other `.md` file. Do not investigate `amdsmi` warnings. Absolutely make sure that no relevant specification is ignored, even when running into errors. After debugging, write down any new gotchas (things that you were unaware of until debugging) in `@code\projects\scaling_bc_to_perfection\-GOTCHAS.md` (don't forget to take into account existing ones). Do not run any tasks in the background.
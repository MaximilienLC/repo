# Experiment Orchestrator

This is a dark-theme GUI.


## Sections

All subsections have their name on top of their contents and their contents boxed.

### Left Side

#### First Level

These all exist at the same height.

#### SubSection `Environment`

Buttons `local` `ginkgo` `rorqual`.
Only one can be selected at a time. The button selected is highlighted.

#### SubSection `Concurrency`

A `Max Running Tasks` text field, a text box to input the number and a `Set` button.
This applies to the currently selected `environment`.

#### Button `Load Tasks`

#### SubSection `Task Queue`

A scrollable list of tasks with 3 columns: `ID`, `Status` and `Command`.

`Status` can be either `Idle`, `Pending`, `Running`, `Completed` or `Failed`.

All tasks in the queue can be selected using `Ctrl+a`.
Pressing `Ctrl` while selecting a task adds it to a multi task selection.
Pressing `Shift` while pressing the up/down arrow adds the up/down task(s) to the selection.
A scroll bar on the right if the task queue is long.

Buttons under the list: `Queue`, `Remove`


### Right Side: `Task Output`

Shows the output of the task selected in the `Task Queue`.

## Behaviour

When a task reaches status `Running`, the following behaviour occurs:

Regardless of the environment, a new `tmux` session is created locally. Its name is simply the task ID, e.g. `53`. The output of that session becomes visible in the `Task Output` section when the task is selected from the `Task Queue`. 


* If the selected environment is `local`, we 
- If the environment is `local`, a new tmux session is created with the name simply being the task ID. In that tmux session,
- If the environment is either `ginkgo` or `rorqual`, a new tmux session is created locally with t

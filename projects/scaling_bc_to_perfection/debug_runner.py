import sys
import time
from pathlib import Path
from orchestrator import TaskStatus, LocalBackend

def run_single_task(task_id, command, workdir):
    print("========================================")
    print(f"Starting Task #{task_id}")
    backend = LocalBackend(task_id)
    backend.start(command, workdir)
    while backend.get_status() == TaskStatus.RUNNING:
        new_output = backend.poll_output()
        for line in new_output:
            print(line)
        time.sleep(1)
    
    # Print any final output
    final_output = backend.poll_output()
    for line in final_output:
        print(line)

    # Print the initial buffer content which might contain setup errors
    if not final_output and not new_output:
        for line in backend._output_buffer:
            print(line)

    print(f"Task #{task_id} finished with status: {backend.get_status().value}")
    print("========================================")

def main():
    # Make workdir relative to this script's location
    script_dir = Path(__file__).parent
    workdir = str((script_dir / "experiment_1").resolve())
    tasks_file = Path(workdir) / "tasks.txt"
    with open(tasks_file, "r") as f:
        commands = [line for line in f.read().splitlines() if line and not line.startswith("#")]
    for i, command in enumerate(commands):
        run_single_task(i + 1, command, workdir)

if __name__ == "__main__":
    main()

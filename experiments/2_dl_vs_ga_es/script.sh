#!/bin/bash

# Common setup variables
WORK_DIR="experiments/2_dl_vs_ga_es/"
VENV_ACTIVATE=". /scratch/mleclei/venv/bin/activate"

# Array of all 54 commands (9 methods × 3 dataset sizes × 2 datasets)
commands=(
    # CartPole experiments with 90% dataset size (GPU 0)
    "python -u main.py --dataset cartpole --method SGD --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_CE --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_F1 --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_CE --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_F1 --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_CE --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_F1 --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_CE --dataset-size 90 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_F1 --dataset-size 90 --gpu 0"

    # CartPole experiments with 30% dataset size (GPU 0)
    "python -u main.py --dataset cartpole --method SGD --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_CE --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_F1 --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_CE --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_F1 --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_CE --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_F1 --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_CE --dataset-size 30 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_F1 --dataset-size 30 --gpu 0"

    # CartPole experiments with 10% dataset size (GPU 0)
    "python -u main.py --dataset cartpole --method SGD --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_CE --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_fixed_F1 --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_CE --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_ga_adaptive_F1 --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_CE --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_fixed_F1 --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_CE --dataset-size 10 --gpu 0"
    "python -u main.py --dataset cartpole --method simple_es_adaptive_F1 --dataset-size 10 --gpu 0"

    # LunarLander experiments with 90% dataset size (GPU 1)
    "python -u main.py --dataset lunarlander --method SGD --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_CE --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_F1 --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_CE --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_F1 --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_CE --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_F1 --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_CE --dataset-size 90 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_F1 --dataset-size 90 --gpu 1"

    # LunarLander experiments with 30% dataset size (GPU 1)
    "python -u main.py --dataset lunarlander --method SGD --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_CE --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_F1 --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_CE --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_F1 --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_CE --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_F1 --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_CE --dataset-size 30 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_F1 --dataset-size 30 --gpu 1"

    # LunarLander experiments with 10% dataset size (GPU 1)
    "python -u main.py --dataset lunarlander --method SGD --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_CE --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_fixed_F1 --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_CE --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_ga_adaptive_F1 --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_CE --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_fixed_F1 --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_CE --dataset-size 10 --gpu 1"
    "python -u main.py --dataset lunarlander --method simple_es_adaptive_F1 --dataset-size 10 --gpu 1"
)

echo "Starting 54 tmux sessions (9 methods × 3 dataset sizes × 2 datasets)..."

# Loop through the commands and create a session for each
for i in "${!commands[@]}"; do
    CMD="${commands[$i]}"

    # Extract the dataset name (matches text after --dataset until the next space)
    DATASET=$(echo "$CMD" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')

    # Extract the method name (matches text after --method until the next space)
    METHOD=$(echo "$CMD" | sed -n 's/.*--method \([^ ]*\).*/\1/p')

    # Extract the dataset size (matches text after --dataset-size until the next space)
    DATASET_SIZE=$(echo "$CMD" | sed -n 's/.*--dataset-size \([^ ]*\).*/\1/p')

    # Construct the Session ID
    SESSION_ID="${DATASET}_${METHOD}_${DATASET_SIZE}pct"

    echo "Launching $SESSION_ID"

    # 1. Create a new detached tmux session named with the arguments
    tmux new-session -d -s "$SESSION_ID"

    # 2. Send the CD command
    tmux send-keys -t "$SESSION_ID" "cd $WORK_DIR" C-m

    # 3. Send the VENV activation command
    tmux send-keys -t "$SESSION_ID" "$VENV_ACTIVATE" C-m

    # 4. Send the specific Python command
    tmux send-keys -t "$SESSION_ID" "$CMD" C-m
done

echo "All sessions launched. Use 'tmux ls' to view them."

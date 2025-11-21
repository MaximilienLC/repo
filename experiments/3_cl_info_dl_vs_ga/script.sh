#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV      Run specific environment(s): cartpole, mountaincar, acrobot, lunarlander, or 'all'"
    echo "                     Can specify multiple environments separated by commas (e.g., cartpole,lunarlander)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all environments (default)"
    echo "  $0 --env cartpole            # Run only CartPole experiments"
    echo "  $0 --env cartpole,acrobot    # Run CartPole and Acrobot experiments"
    echo "  $0 --env all                 # Run all environments (explicit)"
    exit 1
}

# Parse command line arguments
SELECTED_ENVS="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            SELECTED_ENVS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Convert selected envs to lowercase and split by comma
IFS=',' read -ra ENV_ARRAY <<< "$SELECTED_ENVS"

# Function to check if an environment should be run
should_run_env() {
    local env=$1
    if [[ "$SELECTED_ENVS" == "all" ]]; then
        return 0
    fi
    for selected in "${ENV_ARRAY[@]}"; do
        if [[ "$selected" == "$env" ]]; then
            return 0
        fi
    done
    return 1
}

# Declare associative array for commands by environment
declare -A env_commands

env_commands["cartpole"]="
python -u main.py --dataset cartpole --method SGD --gpu 0
python -u main.py --dataset cartpole --method SGD --use-cl-info --gpu 0
python -u main.py --dataset cartpole --method adaptive_ga_CE --gpu 0
python -u main.py --dataset cartpole --method adaptive_ga_CE --use-cl-info --gpu 0
"

env_commands["mountaincar"]="
python -u main.py --dataset mountaincar --method SGD --gpu 0
python -u main.py --dataset mountaincar --method SGD --use-cl-info --gpu 0
python -u main.py --dataset mountaincar --method adaptive_ga_CE --gpu 0
python -u main.py --dataset mountaincar --method adaptive_ga_CE --use-cl-info --gpu 0
"

env_commands["acrobot"]="
python -u main.py --dataset acrobot --method SGD --gpu 1
python -u main.py --dataset acrobot --method SGD --use-cl-info --gpu 1
python -u main.py --dataset acrobot --method adaptive_ga_CE --gpu 1
python -u main.py --dataset acrobot --method adaptive_ga_CE --use-cl-info --gpu 1
"

env_commands["lunarlander"]="
python -u main.py --dataset lunarlander --method SGD --gpu 1
python -u main.py --dataset lunarlander --method SGD --use-cl-info --gpu 1
python -u main.py --dataset lunarlander --method adaptive_ga_CE --gpu 1
python -u main.py --dataset lunarlander --method adaptive_ga_CE --use-cl-info --gpu 1
"

# Build the list of commands to run
commands=()
for env in cartpole mountaincar acrobot lunarlander; do
    if should_run_env "$env"; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                commands+=("$line")
            fi
        done <<< "${env_commands[$env]}"
    fi
done

# Count how many sessions we're launching
num_sessions=${#commands[@]}
echo "Starting $num_sessions tmux sessions for: $SELECTED_ENVS"
echo ""

# Loop through the commands and create a session for each
for i in "${!commands[@]}"; do
    CMD="${commands[$i]}"

    # Extract the dataset name (matches text after --dataset until the next space)
    DATASET=$(echo "$CMD" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')

    # Extract the method name (matches text after --method until the next space)
    METHOD=$(echo "$CMD" | sed -n 's/.*--method \([^ ]*\).*/\1/p')

    # Check if --use-cl-info flag is present
    if echo "$CMD" | grep -q "\-\-use-cl-info"; then
        CL_VARIANT="with_cl"
    else
        CL_VARIANT="no_cl"
    fi

    # Construct the Session ID (e.g., cartpole_SGD_with_cl)
    SESSION_ID="${DATASET}_${METHOD}_${CL_VARIANT}"

    echo "Launching $SESSION_ID"

    # 1. Create a new detached tmux session named with the arguments
    tmux new-session -d -s "$SESSION_ID"

    # 2. Change to the script's directory
    tmux send-keys -t "$SESSION_ID" "cd \"$SCRIPT_DIR\"" C-m

    # 3. Send the specific Python command
    tmux send-keys -t "$SESSION_ID" "$CMD" C-m
done

echo "All sessions launched. Use 'tmux ls' to view them."
echo ""
echo "Useful commands:"
echo "  tmux ls                           # List all sessions"
echo "  tmux attach -t <session_name>     # Attach to a session"
echo "  tmux kill-session -t <session_name> # Kill a specific session"
echo "  tmux kill-server                  # Kill all sessions"

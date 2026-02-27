#!/bin/bash
set -e
MODE=${1:-matsat}
EXAMPLE=${2:-diamond}

# Consume the first two arguments so we can pass any remaining arguments directly
if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" -eq 1 ]; then shift 1; fi

SESSION_NAME="matsat_distributed"

# Get absolute path of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
fi

# Build common args
COMMON="--num_parties 3 --protocol shamir --host localhost --port 5000 --mode $MODE"
if [ -n "$EXAMPLE" ]; then
    COMMON="$COMMON --example $EXAMPLE"
fi
# Append any remaining arguments
if [ "$#" -gt 0 ]; then
    COMMON="$COMMON $@"
fi

# Start the new session specifically in the script directory
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

# Start Alice first, she will compile the graph. We sleep to give her time.
tmux send-keys -t $SESSION_NAME "uv run python3 run.py --role alice --party_id 0 $COMMON" Enter

echo "Waiting for Alice to finish compiling before launching Bob nodes..."
# Sleep for 60 seconds to ensure Alice finishes compiling (varies depending on graph size)
sleep 75

# Bob 1 is party 1
tmux split-window -h -t $SESSION_NAME -c "$SCRIPT_DIR"
tmux send-keys -t $SESSION_NAME "uv run python3 run.py --role bob --party_id 1 $COMMON" Enter

sleep 2

# Bob 2 is party 2
tmux split-window -v -t $SESSION_NAME -c "$SCRIPT_DIR"
tmux send-keys -t $SESSION_NAME "uv run python3 run.py --role bob --party_id 2 $COMMON" Enter

echo "Tmux session \"$SESSION_NAME\" started. Example: $EXAMPLE, Mode: $MODE"
echo "Attach with: tmux attach -t $SESSION_NAME"

#!/bin/bash
set -e
MODE=${1:-matsat}
EXAMPLE=${2:-diamond}

SESSION_NAME="matsat_distributed"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
fi

# Build common args
COMMON="--num_parties 3 --protocol shamir --host localhost --port 5000 --mode $MODE"
if [ -n "$EXAMPLE" ]; then
    COMMON="$COMMON --example $EXAMPLE"
fi

tmux new-session -d -s $SESSION_NAME

# Alice is party 0 (Orchestrator — starts first, generates Shamir prime, compiles program)
tmux send-keys -t $SESSION_NAME "uv run python3 distributed/run.py --role alice --party_id 0 $COMMON" Enter

sleep 3

# Bob 1 is party 1
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "uv run python3 distributed/run.py --role bob --party_id 1 $COMMON" Enter

sleep 1

# Bob 2 is party 2
tmux split-window -v -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "uv run python3 distributed/run.py --role bob --party_id 2 $COMMON" Enter

echo "Tmux session \"$SESSION_NAME\" started. Example: $EXAMPLE, Mode: $MODE"
echo "Attach with: tmux attach -t $SESSION_NAME"

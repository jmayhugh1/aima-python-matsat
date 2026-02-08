#!/bin/bash

# Kill any existing python or MP-SPDZ processes
echo "Cleaning up old processes..."
pkill -f distributed_bayesien_paths.py
pkill -f shamir-party.x

# Wait a moment for ports to clear
sleep 1

# Common arguments
GRID_SIZE=5
PATH_LENGTH=1
INFO_BUDGET=10.0
ITERATIONS=100
SEED=42
PROTOCOL="shamir"
PORT=5000
HOST="localhost"
NUM_PARTIES=3

SESSION="bayes"

# Create new session
tmux new-session -d -s $SESSION

# Rename first window
tmux rename-window -t $SESSION:0 'alice'

# Pane 0: Alice
# Note: We exec bash at the end to keep the pane open
tmux send-keys -t $SESSION:0 "python3 -u distributed_bayesien_paths.py --role alice --party_id 0 --num_parties $NUM_PARTIES --host $HOST --port $PORT --seed $SEED --grid_size $GRID_SIZE --path_length $PATH_LENGTH --info_budget $INFO_BUDGET --iterations $ITERATIONS --protocol $PROTOCOL; echo 'Alice finished. Press Enter to exit.'; read" C-m

# Give Alice a moment to bind
sleep 2

# Split window for Bob 1 (Horizontal split)
tmux split-window -h -t $SESSION:0
# Select layout to evenly distribute panes horizontally
tmux select-layout -t $SESSION:0 even-horizontal

# Pane 1: Bob 1
tmux send-keys -t $SESSION:0.1 "python3 -u distributed_bayesien_paths.py --role bob --party_id 1 --num_parties $NUM_PARTIES --host $HOST --port $PORT --seed $SEED --grid_size $GRID_SIZE --path_length $PATH_LENGTH --info_budget $INFO_BUDGET --iterations $ITERATIONS --protocol $PROTOCOL; echo 'Bob 1 finished. Press Enter to exit.'; read" C-m

# Split window for Bob 2 (from Pane 1, Horizontal split again?)
# To get 3 columns:
# 1. Split Pane 0 horizontally -> [Pane 0] [Pane 1]
# 2. Split Pane 1 horizontally -> [Pane 0] [Pane 1] [Pane 2]
tmux split-window -h -t $SESSION:0.1
tmux select-layout -t $SESSION:0 even-horizontal

# Pane 2: Bob 2
tmux send-keys -t $SESSION:0.2 "python3 -u distributed_bayesien_paths.py --role bob --party_id 2 --num_parties $NUM_PARTIES --host $HOST --port $PORT --seed $SEED --grid_size $GRID_SIZE --path_length $PATH_LENGTH --info_budget $INFO_BUDGET --iterations $ITERATIONS --protocol $PROTOCOL; echo 'Bob 2 finished. Press Enter to exit.'; read" C-m

# Attach to session
tmux attach -t $SESSION

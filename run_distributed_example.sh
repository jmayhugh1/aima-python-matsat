#!/bin/bash

# Kill any existing python or MP-SPDZ processes
echo "Cleaning up old processes..."
pkill -f distributed_bayesien_paths.py
pkill -f shamir-party.x

# Wait a moment for ports to clear
sleep 1

# Common arguments
GRID_SIZE=5
ITERATIONS=100
SEED=42
PROTOCOL="shamir"
PORT=5000
HOST="localhost"
NUM_PARTIES=3

echo "Starting Alice (Party 0)..."
python3 distributed_bayesien_paths.py \
    --role alice \
    --party_id 0 \
    --num_parties $NUM_PARTIES \
    --host $HOST \
    --port $PORT \
    --seed $SEED \
    --grid_size $GRID_SIZE \
    --iterations $ITERATIONS \
    --protocol $PROTOCOL &

# Give Alice a moment to bind the server port
sleep 2

echo "Starting Bob 1 (Party 1)..."
python3 distributed_bayesien_paths.py \
    --role bob \
    --party_id 1 \
    --num_parties $NUM_PARTIES \
    --host $HOST \
    --port $PORT \
    --seed $SEED \
    --grid_size $GRID_SIZE \
    --iterations $ITERATIONS \
    --protocol $PROTOCOL &

echo "Starting Bob 2 (Party 2)..."
python3 distributed_bayesien_paths.py \
    --role bob \
    --party_id 2 \
    --num_parties $NUM_PARTIES \
    --host $HOST \
    --port $PORT \
    --seed $SEED \
    --grid_size $GRID_SIZE \
    --iterations $ITERATIONS \
    --protocol $PROTOCOL &

# Wait for all background jobs to finish
wait
echo "All parties finished."

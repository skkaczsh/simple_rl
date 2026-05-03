#!/bin/bash
# Run multiple training instances in parallel to maximize CPU utilization
# Target: 80% of 20 cores = 16 cores

INSTANCE_COUNT=${1:-16}  # Default 16 instances
CONFIG="configs/train/phase1b_high_util.yaml"
MANIFEST="configs/embodiment/unitree_g1_43dof_sim_v0.json"
BASE_OUTPUT="runs/phase1b_parallel"

echo "Starting $INSTANCE_COUNT parallel training instances..."
echo "Config: $CONFIG"
echo "Manifest: $MANIFEST"
echo "Output base: $BASE_OUTPUT"

for i in $(seq 1 $INSTANCE_COUNT); do
    OUTPUT="${BASE_OUTPUT}_${i}"
    LOG="${BASE_OUTPUT}_${i}_train.log"

    echo "Starting instance $i -> $OUTPUT"
    nohup python3 scripts/run_school_curriculum.py "$CONFIG" "$MANIFEST" "$OUTPUT" > "$LOG" 2>&1 &
done

echo "All $INSTANCE_COUNT instances started."
echo "Monitor with: ps aux | grep run_school_curriculum | grep -v grep | wc -l"

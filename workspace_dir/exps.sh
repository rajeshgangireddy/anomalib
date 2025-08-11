#!/bin/bash

# Ensure conda is available in this shell
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
models=(
    "Cfa"
    "Cflow"
    "Csflow"
    "Dfkde"
    "Dfm"
    "Dinomaly"
    "Draem"
    "Dsr"
    "EfficientAd"
    "Fastflow"
    "Fre"
    "Ganomaly"
    "Padim"
    "Patchcore"
    "ReverseDistillation"
    "Stfpm"
    "Supersimplenet"
    "Uflow"
    "UniNet"
    "VlmAd"
    "WinClip"
)

completed_file="completed_models.txt"

mkdir -p logs

# Create file if it doesn't exist
touch "$completed_file"

total=${#models[@]}
count=0

for m in "${models[@]}"; do
    ((count++))
    if grep -q "^$m$" "$completed_file"; then
        echo "✅ [$count/$total] Skipping $m (already completed)"
        continue
    fi

    echo "===== [$count/$total] Running $m ====="
    start_time=$(date +"%Y-%m-%d %H:%M:%S")

    python workspace_dir/benchmark_all.py \
        --epochs 20 --device xpu --models "$m" \
        > "logs/${m}.out" 2> "logs/${m}.err"

    status=$?

    if [ $status -eq 0 ]; then
        echo "$m" >> "$completed_file"
        echo "✅ $m completed successfully at $start_time"
    else
        echo "❌ $m failed with status $status"
    fi

    echo "⏸ Waiting 2 minutes before next model..."
    sleep 120
done
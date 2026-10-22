#!/usr/bin/env bash

set -euo pipefail

# Spouštět z kořenového adresáře projektu
mkdir -p logs

for seed in 42 123 7 2025 2026; do
    echo "========================================"
    echo "Training model with seed: $seed"
    echo "========================================"

    PYTHONPATH=. python -u trains/train_cnnlstm.py \
        --exp_name cnnlstm_ensemble \
        --seed "$seed" \
        2>&1 | tee "logs/cnnlstm_ensemble_seed${seed}.log"
done

echo "All five models have been trained successfully."
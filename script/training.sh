#!/bin/bash

# Commands to run for DiffuSETS training on ECG datasets
# Configuration file specifies model architecture, training parameters, and dataset paths
# More useful information available with: python DiffuSETS_train.py --help

# Configuration
EXPERIMENT_NAME="diffusets_ecg_training"
CONFIG_FILE="config/diffusets_inference_config.json"

# Print header
echo "=== DiffuSETS ECG Training Pipeline ==="
echo "Starting execution at $(date)"
echo "Configuration file: $CONFIG_FILE"

# Execute training
echo ""
echo "[EXECUTING] Training DiffuSETS model..."
python DiffuSETS_train.py $CONFIG_FILE

echo ""
echo "=== Training completed at $(date) ==="
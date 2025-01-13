#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Cleaning up background processes..."
    kill $(jobs -p) 2>/dev/null
}

# Set trap to call cleanup on EXIT
trap cleanup EXIT

unset CUDA_VISIBLE_DEVICES

nums=1000
batch=1
gpu_ids_1=2
gpu_ids_2=3

python -m test_scripts.batch_generation --exp_type 'all' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation --exp_type 'noc' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation --exp_type 'novae' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation --exp_type 'nonoc' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation_ptbxl --exp_type 'all' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation_ptbxl --exp_type 'noc' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation_ptbxl --exp_type 'novae' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_generation_ptbxl --exp_type 'nonoc' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &

jobs -p

wait
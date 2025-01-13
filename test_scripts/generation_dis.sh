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
gpu_ids_1=3
gpu_ids_2=2

python -m test_scripts.batch_disease_gen --exp_type 'pvc' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'pac' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'lbbb' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'rbbb' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'sn' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'snt' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'snb' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'sna' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'afl' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'af' --gpu_ids=$gpu_ids_1 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'pacing' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'mi' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'st' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'avbi' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
python -m test_scripts.batch_disease_gen --exp_type 'normal' --gpu_ids=$gpu_ids_2 --nums=$nums --batch=$batch &
jobs -p

wait
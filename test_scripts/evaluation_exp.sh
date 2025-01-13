unset CUDA_VISIBLE_DEVICES
gpu_ids=2

python -m test_scripts.evaluation --exp_type 'all' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation --exp_type 'noc' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation --exp_type 'novae' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation --exp_type 'nonoc' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation --exp_type 'all' --gpu_ids=$gpu_ids --ptbxl &
python -m test_scripts.evaluation --exp_type 'noc' --gpu_ids=$gpu_ids --ptbxl &
python -m test_scripts.evaluation --exp_type 'novae' --gpu_ids=$gpu_ids --ptbxl &
python -m test_scripts.evaluation --exp_type 'nonoc' --gpu_ids=$gpu_ids --ptbxl &

wait
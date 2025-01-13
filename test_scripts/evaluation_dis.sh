unset CUDA_VISIBLE_DEVICES
gpu_ids=2

python -m test_scripts.evaluation_disease --exp_type 'pvc' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'pac' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'lbbb' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'rbbb' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'sn' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'snt' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'snb' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'sna' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'afl' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'af' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'pacing' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'mi' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'st' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'avbi' --gpu_ids=$gpu_ids &
python -m test_scripts.evaluation_disease --exp_type 'normal' --gpu_ids=$gpu_ids &

wait
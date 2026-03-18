# Clone using github 

# Source awm
python -m pip install -e source/awm

# To start training - Change 0 and 1 for using GPU1 or GPU2
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_Morph-v0 --num_envs 1024 --headless

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_Morph-v0 --num_envs 1024 --max_iterations 30000 --headless --device cuda:1

# Wheel only ablation
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_WheelsOnly-v0 --num_envs 1024 --headless --device cuda:0

# Leg only ablation
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_LegsOpen-v0 --num_envs 1024 --headless --device cuda:1

# Propioception only
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py --task Template-Awm_ProprioOnly-v0 --num_envs 1024 --max_iterations 30000 --headless --device cuda:1

# To play the trained policy 
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py --task Template-Awm_Morph-v0 --num_envs 100 --checkpoint /home/shashwat/awm_manager/logs/rsl_rl/awm_morph/2026-02-05_12-55-55/model_250.pt

# Eval stairs 
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py --task Template-Awm_StairsEval-v0 --num_envs 4 --checkpoint logs/rsl_rl/awm_morph/<run>/model_best.pt

# To see the tensorboard logs 
tensorboard --logdir logs/rsl_rl/awm_manager

# To see envs name
python scripts/list_envs.py 

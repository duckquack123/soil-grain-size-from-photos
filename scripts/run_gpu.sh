#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=soil_gpu
#SBATCH --output=/csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/logs/gpu_%j.out
# Generic 1-GPU job. usage: sbatch scripts/run_gpu.sh src/<script>.py [args]
source /csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/scripts/env.sh
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"
python "$@"

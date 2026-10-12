#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=soil_feats
#SBATCH --output=/csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/logs/feats_%j.out
# DINOv2 + texture features at 4.6 px/mm.  usage: sbatch scripts/extract_features.sh OUT.npz [--gray] [--degrade]
source /csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
python src/features.py facebook/dinov2-base "${1:-feats_dinov2base.npz}" "${@:2}"

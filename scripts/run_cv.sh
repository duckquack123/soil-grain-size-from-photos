#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=soil_cv
#SBATCH --output=/csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/logs/cv_%j.out
# CPU-only CV / prediction. usage: sbatch scripts/run_cv.sh src/compare_models.py TAG   (FEATS env selects features)
source /csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/scripts/env.sh
python "$@"

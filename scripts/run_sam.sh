#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=00:58:00
#SBATCH --job-name=soil_sam
#SBATCH --output=/csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/logs/sam_%j.out
# SAM ViT-L grain segmentation, 4 shards on 2 GPUs (2 per A30; 3 per GPU OOMs). usage: sbatch scripts/run_sam.sh [PREFIX]
source /csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos/scripts/env.sh
export OMP_NUM_THREADS=5
PREFIX=${1:-grains_shard}
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) python src/sam_grains.py ${PREFIX}$i.npz --shard $i --nshards 4 > logs/sam_${PREFIX}$i.txt 2>&1 &
  sleep 20
done
wait
grep -h saved logs/sam_${PREFIX}?.txt

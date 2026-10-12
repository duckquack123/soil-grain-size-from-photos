# Common environment for all jobs. Rules: partition mtech only, <= 2 GPUs per user.
source /csehome/m25csa007/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export HF_HUB_OFFLINE=1                       # weights are pre-downloaded to ~/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
PROJECT=/csehome/m25csa007/m25csa007/kaggle/soil-grain-size-from-photos
cd "$PROJECT"

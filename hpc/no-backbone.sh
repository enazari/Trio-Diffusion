#!/bin/bash
#SBATCH --job-name=none
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100:4
#SBATCH --mail-type=ALL

# Self-submit: source .env and resubmit via sbatch with account/email
if [ -z "$SLURM_JOB_ID" ]; then
    source "$(dirname "$0")/../.env"
    exec sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
fi

# Configuration settings
CONFIG_NAME="no_backbone_3k"

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

# Navigate to project root (all paths below are relative to project root)
cd ..

# Stage datasets/ and pretrained_models/ to fast local NVMe.
# Originals are NEVER modified — we just copy and redirect via env vars.
echo "Copying datasets to local storage..."
cp -r datasets $SLURM_TMPDIR/datasets
du -sh $SLURM_TMPDIR/datasets/*
export TRIO_CACHE_DIR=$SLURM_TMPDIR/datasets

if [ -d "pretrained_models" ]; then
    echo "Copying pretrained models to local storage..."
    cp -r pretrained_models $SLURM_TMPDIR/pretrained_models
    export TRIO_MODELS_DIR=$SLURM_TMPDIR/pretrained_models
fi
export HF_HUB_OFFLINE=1

# Print storage information before training
echo "Dataset location: $TRIO_CACHE_DIR"
echo "Models location: ${TRIO_MODELS_DIR:-pretrained_models}"
echo "Current working directory (where results will be saved): $(pwd)"

# Check loaded modules
echo "Successfully loaded modules:"
module list

# Activate environment
echo "Activating Python virtual environment..."
source ../trio-env/bin/activate

# Start training (accelerate auto-detects GPUs from SLURM)
echo "Starting training..."
accelerate launch --mixed_precision=fp16 train.py --config $CONFIG_NAME

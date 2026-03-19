#!/bin/bash
#SBATCH --job-name=trio_install
#SBATCH --time=00:45:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# Self-submit: source .env and resubmit via sbatch with account/email
if [ -z "$SLURM_JOB_ID" ]; then
    source "$(dirname "$0")/../.env"
    exec sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
fi

echo "=== TRIO DIFFUSION ENVIRONMENT INSTALLATION ==="

# Step 1: Load required modules
# H100 GPUs require torch >= 2.5.1, which needs StdEnv/2023 + CUDA 12.x
echo "Loading required modules..."
module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

echo "Loaded modules:"
module list

# Step 2: Navigate to project parent (env lives alongside project dir)
# Layout: .../trio/hpc/ (we are here)
#         .../trio-env/  (env goes here)
cd ../..
PERSISTENT_DIR=$(pwd)/trio-env
echo "Current directory: $(pwd)"
echo "Persistent venv will be: $PERSISTENT_DIR"

# Remove existing environment if it exists
if [ -d "$PERSISTENT_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$PERSISTENT_DIR"
fi

# Step 3: Clear PYTHONPATH
echo "Clearing PYTHONPATH to avoid conflicts..."
unset PYTHONPATH
export PYTHONPATH=""

# Step 4: Build venv on SLURM_TMPDIR (fast local NVMe) to avoid Lustre I/O errors
# Large pip installs (torch ~2GB) cause [Errno 14] Bad address on Lustre.
BUILD_ENV=$SLURM_TMPDIR/trio-env
echo "Creating virtual environment in SLURM_TMPDIR..."
python -m venv --system-site-packages $BUILD_ENV

if [ ! -d "$BUILD_ENV" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

source $BUILD_ENV/bin/activate
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set - activation failed"
    exit 1
fi
echo "Build venv activated: $VIRTUAL_ENV"

# Step 5: Upgrade pip
echo "Upgrading pip..."
pip install --no-index --upgrade pip
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

# Step 6: Install packages (all from HPC wheel cache, no internet needed)
# All packages below are confirmed available in the ComputeCanada wheel cache.
# Using --no-index throughout to avoid costly PyPI timeout retries on compute nodes.
# --retries 2 limits pip to 2 attempts per package (down from default 5).

echo "Installing numpy..."
pip install --no-index --retries 2 numpy
if [ $? -ne 0 ]; then echo "ERROR: Failed to install numpy"; exit 1; fi

echo "Installing PyTorch ecosystem..."
pip install --no-index --retries 2 torch torchvision torchaudio
if [ $? -ne 0 ]; then echo "ERROR: Failed to install torch/torchvision/torchaudio"; exit 1; fi

echo "Installing HPC-cached packages..."
pip install --no-index --retries 2 scipy pillow tqdm pyyaml matplotlib psutil
if [ $? -ne 0 ]; then
    echo "WARNING: Some HPC-cached packages failed, continuing..."
fi

echo "Installing lmdb..."
pip install --no-index --retries 2 lmdb
if [ $? -ne 0 ]; then echo "ERROR: Failed to install lmdb"; exit 1; fi

echo "Installing accelerate..."
pip install --no-index --retries 2 accelerate
if [ $? -ne 0 ]; then echo "ERROR: Failed to install accelerate"; exit 1; fi

echo "Installing open-clip-torch (for CLIP global loss)..."
pip install --no-index --retries 2 open-clip-torch
if [ $? -ne 0 ]; then echo "ERROR: Failed to install open-clip-torch"; exit 1; fi

echo "Installing lpips (perceptual loss)..."
pip install --no-index --retries 2 lpips
if [ $? -ne 0 ]; then echo "ERROR: Failed to install lpips"; exit 1; fi

echo "Installing tensorboard..."
pip install --no-index --retries 2 tensorboard
if [ $? -ne 0 ]; then echo "ERROR: Failed to install tensorboard"; exit 1; fi

# Step 7: Verify packages before copying
echo "Verifying package compatibility..."
python -c "
import numpy as np
print(f'NumPy {np.__version__}')

import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')

import torchvision
print(f'Torchvision {torchvision.__version__}')

import torchaudio
print(f'Torchaudio {torchaudio.__version__}')

import scipy
print(f'SciPy {scipy.__version__}')

import lmdb
print(f'lmdb {lmdb.version()}')

import tqdm
print(f'tqdm {tqdm.__version__}')

import matplotlib
print(f'matplotlib {matplotlib.__version__}')

from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
test_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
result = ToTensor()(Image.fromarray(test_array))
print(f'ToTensor transform: {result.shape}')

print('All critical packages working.')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

# Step 8: Copy venv from SLURM_TMPDIR to persistent Lustre storage
deactivate
echo "Copying venv to persistent storage: $PERSISTENT_DIR ..."
cp -a $BUILD_ENV $PERSISTENT_DIR

# Fix hardcoded paths (SLURM_TMPDIR path -> persistent path)
echo "Fixing venv paths..."
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.csh
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.fish
# Fix shebangs in pip and other scripts
find $PERSISTENT_DIR/bin -type f -exec grep -l "$BUILD_ENV" {} + 2>/dev/null | \
    xargs -r sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g"

# Verify the persistent copy works
source $PERSISTENT_DIR/bin/activate
echo "Persistent venv activated: $VIRTUAL_ENV"
python -c "import torch; print(f'torch {torch.__version__} OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: Persistent venv verification failed"
    exit 1
fi

# Step 9: Summary
echo "=== INSTALLATION SUMMARY ==="
echo "Virtual environment: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "=== INSTALLATION COMPLETED ==="
echo ""
echo "To use in training scripts:"
echo "  module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1"
echo "  unset PYTHONPATH && export PYTHONPATH=\"\""
echo "  source ../trio-env/bin/activate"

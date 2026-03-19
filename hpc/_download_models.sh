#!/bin/bash
# Pre-download model weights for offline use on compute nodes.
# Run ONCE on a login node (has internet) before submitting training jobs.
#
# Usage (from hpc/ directory):
#   bash _download_models.sh

# Activate environment
echo "Activating Python virtual environment..."
source ../../trio-env/bin/activate

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

mkdir -p pretrained_models

echo "=== Downloading model weights to pretrained_models/ ==="

# CLIP ViT-B/32 (~340MB)
DST="pretrained_models/clip-vitb32-openai.bin"
if [ ! -f "$DST" ]; then
    echo "Downloading CLIP ViT-B/32..."
    wget -O "$DST" "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/open_clip_pytorch_model.bin"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download CLIP weights"
        rm -f "$DST"
        exit 1
    fi
    echo "  saved: $DST ($(du -h "$DST" | cut -f1))"
else
    echo "CLIP already cached: $DST"
fi

# DINOv2 ViT-B/14 weights (~330MB)
DST="pretrained_models/dinov2-vitb14.pth"
if [ ! -f "$DST" ]; then
    echo "Downloading DINOv2 ViT-B/14..."
    python -c "import torch; torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        '$DST')"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download DINOv2 weights"
        rm -f "$DST"
        exit 1
    fi
    echo "  saved: $DST ($(du -h "$DST" | cut -f1))"
else
    echo "DINOv2 already cached: $DST"
fi

# Cache DINOv2 architecture code (hub repo, needed offline)
mkdir -p pretrained_models/hub
python -c "
import torch
torch.hub.set_dir('pretrained_models/hub')
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False, verbose=False)
print('DINOv2 hub architecture cached.')
"

echo ""
echo "=== Cached files ==="
ls -lh pretrained_models/
echo ""
echo "=== All models cached. Ready for offline compute nodes. ==="

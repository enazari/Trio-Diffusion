"""DINOv2 ViT-B/14 visual encoder for global context conditioning."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_CACHE_DIR = os.environ.get(
    "TRIO_MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "pretrained_models"),
)
_HUB_DIR = os.path.join(_CACHE_DIR, "hub")

_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOBackbone(nn.Module):
    """Frozen DINOv2 ViT-B/14 returning patch tokens for cross-attention.

    Accepts [B, 3, H, W] in [-1, 1] (pipeline convention).
    Internally resizes to 224x224 and applies ImageNet normalization.
    Returns patch tokens [B, 256, 768] (16x16 grid, no CLS).
    """

    def __init__(self):
        super().__init__()
        torch.hub.set_dir(_HUB_DIR)
        local_weights = os.path.join(_CACHE_DIR, "dinov2-vitb14.pth")
        if os.path.exists(local_weights):
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=False, verbose=False,
            )
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=True, verbose=False,
            )
        self.dino = model
        self.dino.requires_grad_(False)
        self.register_buffer("dino_mean", _DINO_MEAN)
        self.register_buffer("dino_std",  _DINO_STD)

    @torch.no_grad()
    def forward(self, x):
        # [-1,1] -> [0,1]
        x = x * 0.5 + 0.5
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # ImageNet normalize
        x = (x - self.dino_mean) / self.dino_std
        # Get patch tokens: [B, 256, 768] (16x16 grid, no CLS token)
        feats = self.dino.forward_features(x)
        return feats["x_norm_patchtokens"]


def build_dino(config):
    return DINOBackbone()

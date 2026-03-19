"""CLIP ViT-B/32 visual + text encoder for global context conditioning."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip import tokenize as clip_tokenize

_CACHE_DIR = os.environ.get(
    "TRIO_MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "pretrained_models"),
)

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _resolve_pretrained(model_name, pretrained):
    """Return local checkpoint path if cached, else return pretrained tag."""
    fname = f"clip-{model_name.lower().replace('-', '')}-{pretrained}.bin"
    local = os.path.join(_CACHE_DIR, fname)
    if os.path.exists(local):
        return local
    return pretrained


class CLIPBackbone(nn.Module):
    """Frozen CLIP visual encoder returning patch tokens for cross-attention.

    Accepts [B, 3, H, W] in [-1, 1] (pipeline convention).
    Internally resizes to 224x224 and applies CLIP normalization.
    Returns patch tokens [B, 49, 768] (7x7 grid, no CLS).
    """

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        resolved = _resolve_pretrained(model_name, pretrained)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=resolved,
        )
        self.visual = model.visual
        self.visual.output_tokens = True  # Return (pooled, tokens) tuple
        self.visual.requires_grad_(False)
        self._clip_model = model  # keep full model for text encoding
        self._clip_model.requires_grad_(False)
        self._num_visual_tokens = 49  # 7x7 grid for ViT-B-32
        self.register_buffer("clip_mean", _CLIP_MEAN)
        self.register_buffer("clip_std", _CLIP_STD)

    @torch.no_grad()
    def forward(self, x):
        # [-1,1] -> [0,1]
        x = x * 0.5 + 0.5
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # CLIP normalize
        x = (x - self.clip_mean) / self.clip_std
        # Get patch tokens: pooled=[B,512], tokens=[B,49,768]
        _pooled, tokens = self.visual(x)
        # Return only patch tokens [B, 49, 768] — pooled is 512-d (proj head), skip it
        return tokens

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text prompt into pseudo visual tokens [1, 49, 768].

        Uses the transpose of the visual projection head to map from
        CLIP's shared 512-d space back to the 768-d patch token space.
        """
        tokens = clip_tokenize([text]).to(self.clip_mean.device)
        # Pooled text features in CLIP's shared space [1, 512]
        text_features = self._clip_model.encode_text(tokens, normalize=True)
        # Map 512 → 768 via transpose of visual projection (768→512)
        visual_proj = self._clip_model.visual.proj  # [768, 512]
        pseudo_tokens = text_features @ visual_proj.T  # [1, 768]
        # Tile to match visual token count for cross-attention
        return pseudo_tokens.unsqueeze(1).expand(-1, self._num_visual_tokens, -1)


def build_clip(config):
    return CLIPBackbone()

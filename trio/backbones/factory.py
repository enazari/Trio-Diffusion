"""Build backbone from config."""

import torch.nn as nn


def build_backbone(config) -> nn.Module | None:
    name = config.backbone.name
    if name == "none":
        return None
    if name == "clip":
        from trio.backbones.clip import build_clip
        return build_clip(config)
    if name == "dino":
        from trio.backbones.dino import build_dino
        return build_dino(config)
    raise ValueError(f"Unknown backbone: {name}")

"""Image processing utilities."""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List, Tuple
import random


def add_top_left_padding(image: torch.Tensor, padding_pixels: int, fill_value: float = 0.0) -> torch.Tensor:
    """Add padding only to top and left sides. [C, H, W] -> [C, H+pad, W+pad]."""
    C, H, W = image.shape
    padded = torch.full((C, H + padding_pixels, W + padding_pixels), fill_value)
    padded[:, padding_pixels:, padding_pixels:] = image
    return padded


def crop_for_perfect_division(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Center-crop image so H and W are divisible by patch_size."""
    C, H, W = image.shape
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size
    start_H = (H - new_H) // 2
    start_W = (W - new_W) // 2
    return image[:, start_H:start_H + new_H, start_W:start_W + new_W]


def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extract non-overlapping patches. [C, H, W] -> [N, C, ps, ps]."""
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches_H = H // patch_size
    patches_W = W // patch_size
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, patches_H, patches_W, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, C, patch_size, patch_size)


def extract_flexible_context_trio(
    image: torch.Tensor, trio_block_size: int, row: int, col: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract context trio (TL, TR, BL) and target (BR) from a 2x2 block position."""
    C, H, W = image.shape
    y = row * trio_block_size
    x = col * trio_block_size
    if y + 2 * trio_block_size > H or x + 2 * trio_block_size > W:
        raise ValueError(f"Trio block at ({row}, {col}) exceeds image bounds")

    tl = image[:, y:y + trio_block_size, x:x + trio_block_size]
    tr = image[:, y:y + trio_block_size, x + trio_block_size:x + 2 * trio_block_size]
    bl = image[:, y + trio_block_size:y + 2 * trio_block_size, x:x + trio_block_size]
    br = image[:, y + trio_block_size:y + 2 * trio_block_size, x + trio_block_size:x + 2 * trio_block_size]
    context_trio = torch.cat([tl, tr, bl], dim=0)
    return context_trio, br


def tensor_to_pil(tensor: torch.Tensor, normalize_range: Tuple[float, float] = (-1, 1)) -> Image.Image:
    """Convert tensor [C, H, W] to PIL Image, denormalizing from normalize_range."""
    tensor = tensor.clone()
    tensor = (tensor - normalize_range[0]) / (normalize_range[1] - normalize_range[0])
    tensor = torch.clamp(tensor, 0, 1)
    return TF.to_pil_image(tensor)


def save_comparison_image(original: torch.Tensor, generated: torch.Tensor,
                         save_path: str, normalize_range: Tuple[float, float] = (-1, 1)) -> None:
    """Save side-by-side comparison of original and generated images."""
    def denorm(img):
        img = img.clone()
        img = (img - normalize_range[0]) / (normalize_range[1] - normalize_range[0])
        return torch.clamp(img, 0, 1)

    original_pil = TF.to_pil_image(denorm(original))
    generated_pil = TF.to_pil_image(denorm(generated))
    w, h = original_pil.size
    comparison = Image.new('RGB', (w * 2, h))
    comparison.paste(original_pil, (0, 0))
    comparison.paste(generated_pil, (w, 0))
    comparison.save(save_path)


def _diversity_filter_samples(
    image: torch.Tensor, patch_size: int, num_samples: int,
    min_diversity: float = 0.05, padding_patches: int = 1,
    enable_edge_sampling: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
    """Sample 2x2 blocks with diversity filtering. Returns (context_trio, target, position)."""
    C, H, W = image.shape
    patches_H = H // patch_size
    patches_W = W // patch_size
    patches = extract_patches(image, patch_size).view(patches_H, patches_W, C, patch_size, patch_size)

    samples = []
    max_top = patches_H - 2
    max_left = patches_W - 2

    for _ in range(num_samples * 3):
        if len(samples) >= num_samples:
            break
        r = random.randint(0, max_top)
        c = random.randint(0, max_left)
        tl, tr, bl, br = patches[r, c], patches[r, c+1], patches[r+1, c], patches[r+1, c+1]

        is_grey = lambda p: torch.var(p).item() < 0.001
        if not (enable_edge_sampling and any(is_grey(p) for p in [tl, tr, bl, br])):
            min_mse = min(
                torch.nn.functional.mse_loss(tl, tr).item(),
                torch.nn.functional.mse_loss(tl, bl).item(),
                torch.nn.functional.mse_loss(tr, br).item(),
                torch.nn.functional.mse_loss(bl, br).item(),
            )
            if min_mse < min_diversity:
                continue

        context_trio = torch.cat([tl, tr, bl], dim=0)
        samples.append((context_trio, br, (r + 1, c + 1)))

    return samples


def get_context_trio_samples(image: torch.Tensor, patch_size: int, num_samples: int = 50,
                            padding_pixels: int = 64, enable_edge_sampling: bool = True
                            ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate (context_trio, target_patch) samples from an image."""
    samples = get_context_trio_samples_with_positions(
        image, patch_size, num_samples, padding_pixels, enable_edge_sampling
    )
    return [(ctx, tgt) for ctx, tgt, _ in samples]


def get_context_trio_samples_with_positions(
    image: torch.Tensor, patch_size: int, num_samples: int = 50,
    padding_pixels: int = 64, enable_edge_sampling: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
    """Generate (context_trio, target_patch, position) samples from an image."""
    padding_patches = padding_pixels // patch_size if enable_edge_sampling else 0
    return _diversity_filter_samples(
        image, patch_size, num_samples,
        padding_patches=padding_patches, enable_edge_sampling=enable_edge_sampling
    )


def apply_generated_patches(original_image: torch.Tensor, generated_patches: List[torch.Tensor],
                           positions: List[Tuple[int, int]], patch_size: int) -> torch.Tensor:
    """Apply generated patches at given (row, col) positions."""
    result = original_image.clone()
    for patch, (row, col) in zip(generated_patches, positions):
        h, w = row * patch_size, col * patch_size
        result[:, h:h + patch_size, w:w + patch_size] = patch
    return result


def apply_flexible_generated_patches(
    original_image: torch.Tensor, generated_patches: List[torch.Tensor],
    positions: List[Tuple[int, int]], trio_block_size: int,
) -> torch.Tensor:
    """Apply generated patches at trio block positions (BR of 2x2 block)."""
    result = original_image.clone()
    for patch, (row, col) in zip(generated_patches, positions):
        h = row * trio_block_size + trio_block_size
        w = col * trio_block_size + trio_block_size
        if h + trio_block_size <= result.shape[1] and w + trio_block_size <= result.shape[2]:
            result[:, h:h + trio_block_size, w:w + trio_block_size] = patch
    return result

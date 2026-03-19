"""
Simple and robust image loading utilities for the Trio pipeline.
"""

import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import List, Tuple, Optional
import random

from ..utils.image_utils import crop_for_perfect_division, add_top_left_padding


class ImageLoader:
    """Simple, robust image loader with single responsibility: load and preprocess images."""
    
    def __init__(
        self,
        dataset_path: str,
        patch_size: int = 64,
        step_size: int = 64,
        padding_pixels: int = 64,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        min_image_size: int = 256,
        enable_edge_sampling: bool = True,
        max_images: Optional[int] = None,
        skip_images: int = 0,
        lmdb_path: Optional[str] = None
    ):
        """
        Initialize the image loader.

        Args:
            dataset_path: Path to directory containing images
            patch_size: Size of patches to extract
            step_size: Step size for patch extraction
            padding_pixels: Padding to add around images (only used if enable_edge_sampling=True)
            normalize_range: Range to normalize images to
            min_image_size: Minimum image size to consider
            enable_edge_sampling: Whether to enable edge sampling with top-left padding
            max_images: Maximum number of images to load (None for all)
            skip_images: Number of images to skip from the beginning (for train/eval split)
            lmdb_path: Path to LMDB cache. If set, reads from LMDB instead of disk.
        """
        self.dataset_path = Path(dataset_path)
        self.patch_size = patch_size
        self.step_size = step_size
        self.padding_pixels = padding_pixels
        self.normalize_range = normalize_range
        self.min_image_size = min_image_size
        self.enable_edge_sampling = enable_edge_sampling
        self._lmdb_env = None
        self._lmdb_path = None
        self._lmdb_pid = None

        if lmdb_path and Path(lmdb_path).exists():
            self._init_from_lmdb(lmdb_path, skip_images, max_images)
        else:
            self.image_paths = self._discover_images(max_images, skip_images)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"📁 Found {len(self.image_paths)} valid images in {dataset_path}")
    
    def _init_from_lmdb(self, lmdb_path: str, skip_images: int, max_images: Optional[int]):
        """Initialize from LMDB cache instead of discovering images from disk.

        The LMDB env is NOT kept open here — only metadata is read, then
        the env is closed.  Each DataLoader worker will lazily open its
        own env via _get_lmdb_env() after fork, avoiding LMDB's
        fork-safety issues (shared mmap + locks → segfault).
        """
        import lmdb as _lmdb
        import json

        self._lmdb_path = str(lmdb_path)
        env = _lmdb.open(self._lmdb_path, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                meta = json.loads(txn.get(b"__meta__").decode())
        finally:
            env.close()

        total = meta["num_images"]
        self._lmdb_start = skip_images
        end = total if max_images is None else min(skip_images + max_images, total)
        self._lmdb_count = max(end - skip_images, 0)
        self.image_paths = []  # not needed for LMDB

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            label = "unseen eval" if skip_images > 0 else "training"
            print(f"📁 LMDB ({label}): {self._lmdb_count} images from {lmdb_path}")

    def _get_lmdb_env(self):
        """Lazily open the LMDB env, reopening after fork (PID change)."""
        pid = os.getpid()
        if self._lmdb_env is None or self._lmdb_pid != pid:
            import lmdb as _lmdb
            self._lmdb_env = _lmdb.open(self._lmdb_path, readonly=True, lock=False)
            self._lmdb_pid = pid
        return self._lmdb_env

    def _read_from_lmdb(self, idx: int) -> torch.Tensor:
        """Read a preprocessed image tensor from LMDB."""
        import numpy as np

        actual_idx = self._lmdb_start + idx
        with self._get_lmdb_env().begin() as txn:
            data = txn.get(str(actual_idx).encode())
        if data is None:
            raise IndexError(f"LMDB index {actual_idx} not found")

        shape = np.frombuffer(data[:24], dtype=np.int64)
        arr = np.frombuffer(data[24:], dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy())

    def _discover_images(self, max_images: Optional[int], skip_images: int = 0) -> List[Path]:
        """Discover and filter valid image files.

        Args:
            max_images: Maximum number of images to return (None for all)
            skip_images: Number of valid images to skip from the beginning
        """
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.dataset_path.glob(f"*{ext}"))
            image_paths.extend(self.dataset_path.glob(f"*{ext.upper()}"))

        # Filter by size, sort for deterministic ordering, then slice
        valid_paths = sorted(
            [p for p in image_paths if self._is_valid_image(p)],
            key=lambda p: p.name
        )
        valid_paths = valid_paths[skip_images:]
        if max_images:
            valid_paths = valid_paths[:max_images]

        return valid_paths
    
    def _is_valid_image(self, path: Path) -> bool:
        """Check if image is valid and meets size requirements."""
        try:
            with Image.open(path) as img:
                return min(img.size) >= self.min_image_size
        except Exception:
            return False
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to target range."""
        if self.normalize_range == (-1.0, 1.0):
            return tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        elif self.normalize_range == (0.0, 1.0):
            return tensor  # Already [0,1]
        else:
            min_val, max_val = self.normalize_range
            return tensor * (max_val - min_val) + min_val
    
    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor [C, H, W] ready for patch extraction
        """
        # Load image
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to tensor [0,1]
            tensor = TF.to_tensor(img)
        
        # Handle invalid values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Resize if too large (memory efficiency)
        _, H, W = tensor.shape
        max_size = 1024
        if H > max_size or W > max_size:
            scale = max_size / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            tensor = TF.resize(tensor, (new_H, new_W), antialias=True)
        
        # Normalize to target range
        tensor = self._normalize_tensor(tensor)
        
        # Crop for perfect patch division
        cropped = crop_for_perfect_division(tensor, self.patch_size)
        
        # Add padding if edge sampling enabled
        if self.enable_edge_sampling:
            return add_top_left_padding(cropped, self.padding_pixels, fill_value=0.0)
        else:
            return cropped
    
    def load_images(self, indices: List[int]) -> List[torch.Tensor]:
        """Load multiple images by their indices."""
        images = []
        for idx in indices:
            if 0 <= idx < len(self.image_paths):
                try:
                    image = self.load_image(self.image_paths[idx])
                    images.append(image)
                except Exception as e:
                    print(f"⚠️ Failed to load image {self.image_paths[idx]}: {e}")
                    continue
        return images
    
    def get_random_indices(self, count: int) -> List[int]:
        """Get random image indices."""
        return random.sample(range(len(self.image_paths)), min(count, len(self.image_paths)))
    
    def __len__(self) -> int:
        """Return number of available images."""
        if self._lmdb_path is not None:
            return self._lmdb_count
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load image by index (from LMDB if available, otherwise from disk)."""
        if self._lmdb_path is not None:
            if not 0 <= idx < self._lmdb_count:
                raise IndexError(f"Index {idx} out of range [0, {self._lmdb_count})")
            return self._read_from_lmdb(idx)
        if not 0 <= idx < len(self.image_paths):
            raise IndexError(f"Index {idx} out of range [0, {len(self.image_paths)})")
        return self.load_image(self.image_paths[idx])
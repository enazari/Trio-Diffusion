"""
LMDB cache for preprocessed images.

Stores preprocessed image tensors in a memory-mapped LMDB file for fast loading.
First run builds the cache; subsequent runs load from it instantly.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from .image_loader import ImageLoader


NUM_UNSEEN_EVAL = 50


def get_lmdb_path(cache_dir: str, total_images: int) -> Path:
    """Get LMDB path based on total image count."""
    return Path(cache_dir) / f"trio_{total_images}.lmdb"


def _build_match_config(trio_block_size, normalize_range, enable_edge_sampling):
    """Preprocessing params that determine LMDB compatibility.

    dataset_path is intentionally excluded: pre-built LMDBs transferred
    between machines (e.g., local -> HPC) would always mismatch on path,
    triggering a destructive rebuild even though the data is valid.

    trio_block_size is also excluded: it does NOT affect image preprocessing
    (what gets stored in LMDB). It only affects patch sampling during training.
    Different block sizes can safely use the same preprocessed images.
    """
    return {
        "normalize_range": list(normalize_range),
        "enable_edge_sampling": enable_edge_sampling,
    }


def _build_full_metadata(dataset_path, trio_block_size, normalize_range, enable_edge_sampling):
    """Full metadata stored in LMDB (includes dataset_path for provenance)."""
    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        **_build_match_config(trio_block_size, normalize_range, enable_edge_sampling),
    }


def _read_metadata(lmdb_path: Path) -> dict:
    """Read metadata from an existing LMDB."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            raw = txn.get(b"__meta__")
            if raw is None:
                return {}
            return json.loads(raw.decode())
    finally:
        env.close()


def _config_matches(meta: dict, config: dict) -> bool:
    """Check if LMDB metadata matches current config."""
    for key in config:
        if meta.get(key) != config[key]:
            return False
    return True


def _build_lmdb(
    lmdb_path: Path,
    dataset_path: str,
    total_images: int,
    trio_block_size: int,
    normalize_range: Tuple[float, float],
    enable_edge_sampling: bool,
):
    """Build LMDB cache from source images."""
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building LMDB cache: {lmdb_path}")
    print(f"  Source: {dataset_path}")
    print(f"  Images: {total_images} (training + {NUM_UNSEEN_EVAL} unseen eval)")

    # Create ImageLoader to discover and preprocess images
    loader = ImageLoader(
        dataset_path=dataset_path,
        patch_size=trio_block_size,
        step_size=trio_block_size,
        padding_pixels=trio_block_size if enable_edge_sampling else 0,
        normalize_range=normalize_range,
        enable_edge_sampling=enable_edge_sampling,
        max_images=total_images,
    )

    actual_count = len(loader)
    if actual_count < total_images:
        print(f"  Warning: only {actual_count} valid images found (requested {total_images})")

    # Estimate map size: ~12MB per image (generous for 1024x1024 float32 RGB)
    map_size = max(actual_count * 12 * 1024 * 1024, 100 * 1024 * 1024)

    env = lmdb.open(str(lmdb_path), map_size=map_size)
    try:
        with env.begin(write=True) as txn:
            shapes = []
            for i in tqdm(range(actual_count), desc="Caching images"):
                tensor = loader[i]
                arr = tensor.numpy()
                shape = np.array(arr.shape, dtype=np.int64)
                data = shape.tobytes() + arr.tobytes()
                txn.put(str(i).encode(), data)
                shapes.append(list(arr.shape))

            # Store metadata
            full_meta = _build_full_metadata(
                dataset_path, trio_block_size, normalize_range, enable_edge_sampling
            )
            meta = {
                **full_meta,
                "num_images": actual_count,
                "shapes": shapes,
            }
            txn.put(b"__meta__", json.dumps(meta).encode())
    finally:
        env.close()

    print(f"  LMDB cache built: {actual_count} images")


def ensure_lmdb(
    cache_dir: str,
    dataset_path: str,
    total_images: int,
    trio_block_size: int,
    normalize_range: Tuple[float, float],
    enable_edge_sampling: bool,
) -> Path:
    """
    Ensure an LMDB cache exists with the right config. Build if needed.

    Args:
        cache_dir: Directory to store LMDB files (e.g., "datasets")
        dataset_path: Source image directory
        total_images: Total images to cache (training + unseen eval)
        trio_block_size: Block size for preprocessing
        normalize_range: Normalization range
        enable_edge_sampling: Whether edge sampling padding is enabled

    Returns:
        Path to the LMDB cache
    """
    path = get_lmdb_path(cache_dir, total_images)
    match_config = _build_match_config(
        trio_block_size, normalize_range, enable_edge_sampling
    )

    # Only rank 0 should build/rebuild to avoid DDP race conditions
    is_main_rank = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if is_main_rank:
        if path.exists() or path.is_symlink():
            try:
                meta = _read_metadata(path)
                if _config_matches(meta, match_config) and meta.get("num_images", 0) >= total_images:
                    print(f"Loading from LMDB cache: {path} ({meta['num_images']} images)")
                    return path
                else:
                    print(f"LMDB config mismatch — rebuilding: {path}")
                    if path.is_symlink():
                        path.unlink()
                    else:
                        shutil.rmtree(str(path))
            except Exception as e:
                print(f"LMDB corrupted ({e}) — rebuilding: {path}")
                if path.is_symlink():
                    path.unlink()
                else:
                    shutil.rmtree(str(path), ignore_errors=True)

        _build_lmdb(
            path, dataset_path, total_images, trio_block_size, normalize_range, enable_edge_sampling
        )
    else:
        # Non-main ranks wait for rank 0 to finish building
        import time
        max_wait = 600  # 10 minutes
        elapsed = 0
        while not (path.exists() or path.is_symlink()):
            time.sleep(0.5)
            elapsed += 0.5
            if elapsed > max_wait:
                raise TimeoutError(f"LMDB cache {path} not built after {max_wait}s")

    return path


def read_image_from_lmdb(env: lmdb.Environment, idx: int) -> torch.Tensor:
    """Read a preprocessed image tensor from LMDB.

    Args:
        env: Open LMDB environment (readonly)
        idx: Image index in the LMDB

    Returns:
        Image tensor [3, H, W]
    """
    with env.begin() as txn:
        data = txn.get(str(idx).encode())
    if data is None:
        raise IndexError(f"Image index {idx} not found in LMDB")

    shape = np.frombuffer(data[:24], dtype=np.int64)
    arr = np.frombuffer(data[24:], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())

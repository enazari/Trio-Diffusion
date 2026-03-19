"""
Minimal on-the-fly patch dataset for training the Trio diffusion model.

This dataset generates context trios dynamically for each minibatch,
eliminating RAM bottlenecks by not storing patches in memory.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import random
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm

from .image_loader import ImageLoader
from ..utils.image_utils import extract_patches, extract_flexible_context_trio


def _is_main_process():
    """Check if this is the main process (rank 0 or non-distributed)."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


class PatchDataset(Dataset):
    """
    Minimal dataset that generates context trios on-the-fly for each minibatch.
    
    No RAM bottlenecks - images are loaded and processed only when needed.
    """
    
    def __init__(
        self,
        dataset_path: str,
        patch_size: int = 64,
        trio_block_size: int = None,
        step_size: int = 64,
        padding_pixels: int = 64,
        samples_per_epoch: int = 50000,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        enable_edge_sampling: bool = True,
        max_images: int = None,
        num_images: int = None,
        return_source_image: bool = False,
        lmdb_path: str = None,
        eval_samples_per_image: int = None,
    ):
        """
        Initialize the patch dataset with flexible trio sizing.

        Args:
            dataset_path: Path to directory containing images
            patch_size: DEPRECATED - Size of each patch (for backward compatibility)
            trio_block_size: Size of each context patch and the 2x2 block
            step_size: Step size for sampling patches (allows overlapping when < patch_size)
            padding_pixels: Padding for edge sampling
            samples_per_epoch: Total samples per epoch (determines dataset "length")
            normalize_range: Range to normalize images to
            enable_edge_sampling: Whether to enable edge sampling with padding
            max_images: Maximum number of images to load
            num_images: Alias for max_images (for config compatibility)
            lmdb_path: Path to LMDB cache (if available, loads from LMDB instead of disk)
            eval_samples_per_image: IGNORED during training - only used for evaluation
        """
        # Handle sizing configuration
        if trio_block_size is None:
            trio_block_size = patch_size  # Backward compatibility

        self.trio_block_size = trio_block_size
        self.patch_size = patch_size  # Keep for backward compatibility
        self.step_size = step_size
        self.samples_per_epoch = samples_per_epoch
        self.return_source_image = return_source_image
        self.lmdb_path = lmdb_path

        # Handle num_images alias for config compatibility
        if num_images is not None and max_images is None:
            max_images = num_images

        # Initialize image loader
        self.image_loader = ImageLoader(
            dataset_path=dataset_path,
            patch_size=trio_block_size,
            step_size=step_size,
            padding_pixels=padding_pixels,
            normalize_range=normalize_range,
            enable_edge_sampling=enable_edge_sampling,
            max_images=max_images,
            lmdb_path=lmdb_path,
        )
        
        # Store padding info for flip-augmentation guard
        self.padding_pixels = self.image_loader.padding_pixels
        self.enable_edge_sampling = self.image_loader.enable_edge_sampling

        # Calculate total possible trios from all images
        self.total_possible_trios = self._calculate_total_trios()

        if _is_main_process():
            print(f"🎯 PatchDataset initialized:")
            print(f"   📁 {len(self.image_loader)} images available")
            print(f"   🔲 Patch size: {trio_block_size}x{trio_block_size}")
            print(f"   📊 Samples per epoch: {samples_per_epoch}")
            print(f"   🎲 Total possible trios: {self.total_possible_trios:,}")
            print(f"   💾 RAM usage: minimal (no patch caching)")

            # Save dataset statistics
            self._save_dataset_stats()

            # Print efficiency information
            self._print_efficiency_info()
    
    def _extract_random_trio_from_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """
        Extract a random context trio and target patch from an image.

        Uses random sub-grid offset to break fixed grid alignment, and
        random horizontal/vertical flips of the 2x2 block for augmentation.

        Returns:
            Tuple of (context_trio, target_patch, row, col, grid_h, grid_w)
        """
        C, H, W = image.shape
        trio_block_size = self.trio_block_size

        # Random sub-grid offset to break grid alignment
        dy = random.randint(0, trio_block_size - 1)
        dx = random.randint(0, trio_block_size - 1)
        image = image[:, dy:, dx:]
        C, H, W = image.shape

        trio_blocks_H = H // trio_block_size - 1
        trio_blocks_W = W // trio_block_size - 1

        if trio_blocks_H <= 0 or trio_blocks_W <= 0:
            dummy_patch = torch.zeros(3, trio_block_size, trio_block_size)
            context_trio = torch.cat([dummy_patch, dummy_patch, dummy_patch], dim=0)
            return context_trio, dummy_patch.clone(), 0, 0, 1, 1

        row = random.randint(0, trio_blocks_H - 1)
        col = random.randint(0, trio_blocks_W - 1)

        # Check if this 2x2 block overlaps the top-left padding region.
        # After the random offset crop, some padding may remain.
        remaining_pad_top = max(0, self.padding_pixels - dy) if self.enable_edge_sampling else 0
        remaining_pad_left = max(0, self.padding_pixels - dx) if self.enable_edge_sampling else 0
        block_top = row * trio_block_size
        block_left = col * trio_block_size
        overlaps_padding = (block_top < remaining_pad_top) or (block_left < remaining_pad_left)

        try:
            context_trio, target_patch = extract_flexible_context_trio(
                image, trio_block_size, row, col
            )

            tl, tr, bl = context_trio[0:3], context_trio[3:6], context_trio[6:9]
            br = target_patch

            # Skip flip augmentation when the 2x2 block overlaps the top-left
            # padding region. Padding (gray fill) is added only on top and left
            # sides (see add_top_left_padding), so context patches near those
            # edges contain gray fill pixels. Flipping could swap a
            # gray-containing context patch into the target (BR) position,
            # producing a partially-gray target — something that never happens
            # during autoregressive inference (train/inference mismatch).
            if not overlaps_padding:
                if random.random() < 0.5:  # horizontal flip of 2x2 block
                    tl, tr = torch.flip(tr, [-1]), torch.flip(tl, [-1])
                    bl, br = torch.flip(br, [-1]), torch.flip(bl, [-1])

                if random.random() < 0.5:  # vertical flip of 2x2 block
                    tl, bl = torch.flip(bl, [-2]), torch.flip(tl, [-2])
                    tr, br = torch.flip(br, [-2]), torch.flip(tr, [-2])

            # Context patch dropout: for 1/3 of samples, zero out 1, 2, or all 3
            # context patches (chosen uniformly). Teaches the model to generate
            # from partial or no local context, matching edge/corner conditions
            # during autoregressive generation.
            if random.random() < 1 / 3:
                patches = [tl, tr, bl]
                n_drop = random.randint(1, 3)
                for i in random.sample(range(3), n_drop):
                    patches[i] = torch.zeros_like(patches[i])
                tl, tr, bl = patches

            context_trio = torch.cat([tl, tr, bl], dim=0)
            target_patch = br

            return context_trio, target_patch, row, col, trio_blocks_H, trio_blocks_W
        except Exception:
            dummy_patch = torch.zeros(3, trio_block_size, trio_block_size)
            context_trio = torch.cat([dummy_patch, dummy_patch, dummy_patch], dim=0)
            return context_trio, dummy_patch.clone(), row, col, trio_blocks_H, trio_blocks_W
    
    def _get_random_sample(self) -> Dict[str, torch.Tensor]:
        """Get a random context trio sample from a size-weighted random image."""
        image_idx = random.choices(range(len(self.image_loader)), weights=self.image_weights, k=1)[0]

        try:
            image = self.image_loader[image_idx]
            context_trio, target_patch, row, col, grid_h, grid_w = self._extract_random_trio_from_image(image)

            sample = {
                'context_trio': context_trio,
                'target_patch': target_patch,
                'patch_row': torch.tensor(row, dtype=torch.long),
                'patch_col': torch.tensor(col, dtype=torch.long),
                'grid_h': torch.tensor(grid_h, dtype=torch.long),
                'grid_w': torch.tensor(grid_w, dtype=torch.long),
            }

            if self.return_source_image:
                # Resize full image to 224x224 for backbone encoding
                source = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)
                sample['source_image'] = source

            return sample

        except Exception as e:
            print(f"Failed to load image {image_idx}: {e}")
            return self._get_random_sample()
    
    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Note: idx is ignored - we always return a random sample.
        This ensures good randomization across epochs.
            
        Returns:
            Dictionary with 'context_trio' and 'target_patch' tensors
        """
        return self._get_random_sample()

    def get_fixed_eval_samples(self, num_images: int = 5) -> List[Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]]]:
        """
        Get fixed evaluation samples from training images.

        Args:
            num_images: Number of images to use for evaluation

        Returns:
            List of (image, samples) tuples where samples are (context_trio, target_patch, position) tuples
        """
        eval_dataset = EvaluationDataset(
            dataset_path=str(self.image_loader.dataset_path),
            patch_size=self.patch_size,
            trio_block_size=self.trio_block_size,
            step_size=self.step_size,
            padding_pixels=self.image_loader.padding_pixels,
            normalize_range=self.image_loader.normalize_range,
            enable_edge_sampling=self.image_loader.enable_edge_sampling,
            num_eval_images=num_images,
            lmdb_path=self.lmdb_path,
        )
        return eval_dataset.get_eval_samples(max_samples_per_image=50)

    def get_unseen_eval_samples(self, num_images: int = 5) -> List[Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]]]:
        """
        Get evaluation samples from unseen images (not used in training).

        Skips past all training images and picks from the remainder of the dataset.

        Args:
            num_images: Number of unseen images to use for evaluation

        Returns:
            List of (image, samples) tuples where samples are (context_trio, target_patch, position) tuples
        """
        eval_dataset = EvaluationDataset(
            dataset_path=str(self.image_loader.dataset_path),
            patch_size=self.patch_size,
            trio_block_size=self.trio_block_size,
            step_size=self.step_size,
            padding_pixels=self.image_loader.padding_pixels,
            normalize_range=self.image_loader.normalize_range,
            enable_edge_sampling=self.image_loader.enable_edge_sampling,
            num_eval_images=num_images,
            skip_images=len(self.image_loader),
            lmdb_path=self.lmdb_path,
        )
        return eval_dataset.get_eval_samples(max_samples_per_image=50)

    def _calculate_total_trios(self) -> int:
        """Calculate total number of context trios possible from all images.

        Also builds self.image_weights for size-weighted sampling.
        """
        verbose = _is_main_process()
        if verbose:
            print("🔢 Calculating total possible context trios...")
        total_trios = 0
        failed_images = 0
        self.image_weights = []

        for i in tqdm(range(len(self.image_loader)), desc="Analyzing images", disable=not verbose):
            try:
                # Load image to get dimensions
                image = self.image_loader[i]
                C, H, W = image.shape

                # Calculate patches grid based on step_size
                patches_H = (H - self.patch_size) // self.step_size + 1
                patches_W = (W - self.patch_size) // self.step_size + 1

                # Count possible 2x2 blocks (each forms one trio)
                if patches_H >= 2 and patches_W >= 2:
                    image_trios = (patches_H - 1) * (patches_W - 1)
                    total_trios += image_trios
                    self.image_weights.append(image_trios)
                else:
                    self.image_weights.append(1)

            except Exception as e:
                failed_images += 1
                self.image_weights.append(1)
                print(f"⚠️ Failed to analyze image {i}: {e}")
                continue

        if failed_images > 0:
            print(f"⚠️ Failed to analyze {failed_images} images")
        
        return total_trios
    
    def _save_dataset_stats(self):
        """Save dataset statistics to a file."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.image_loader.dataset_path),
            "total_images": len(self.image_loader),
            "patch_size": self.patch_size,
            "padding_pixels": self.image_loader.padding_pixels,
            "samples_per_epoch": self.samples_per_epoch,
            "total_possible_trios": self.total_possible_trios,
            "enable_edge_sampling": self.image_loader.enable_edge_sampling,
            "normalize_range": self.image_loader.normalize_range,
            "trios_per_image_avg": self.total_possible_trios / max(len(self.image_loader), 1)
        }
        
        # Save to dataset directory
        stats_file = Path(self.image_loader.dataset_path) / "dataset_trio_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Dataset statistics saved to: {stats_file}")
        except Exception as e:
            print(f"⚠️ Failed to save statistics: {e}")
    
    def get_dataset_efficiency(self) -> dict:
        """Get information about dataset usage efficiency."""
        if self.total_possible_trios == 0:
            return {
                "total_possible_trios": 0,
                "samples_per_epoch": self.samples_per_epoch,
                "epochs_to_exhaust_all_trios": 0,
                "dataset_efficiency_per_epoch": 0.0,
                "recommendation": "no valid images found"
            }
        
        epochs_to_exhaust = self.total_possible_trios / self.samples_per_epoch
        efficiency = min(1.0, self.samples_per_epoch / self.total_possible_trios)
        
        return {
            "total_possible_trios": self.total_possible_trios,
            "samples_per_epoch": self.samples_per_epoch,
            "epochs_to_exhaust_all_trios": epochs_to_exhaust,
            "dataset_efficiency_per_epoch": efficiency,
            "recommendation": "increase samples_per_epoch" if efficiency < 0.1 else "good efficiency"
        }

    def _print_efficiency_info(self):
        """Print efficiency information about the dataset."""
        efficiency_info = self.get_dataset_efficiency()
        print(f"🚀 Efficiency Information:")
        print(f"   🎲 Total possible trios: {efficiency_info['total_possible_trios']:,}")
        print(f"   📊 Samples per epoch: {efficiency_info['samples_per_epoch']:,}")
        print(f"   🕒 Epochs to exhaust all trios: {efficiency_info['epochs_to_exhaust_all_trios']:.2f}")
        print(f"   🎯 Dataset efficiency per epoch: {efficiency_info['dataset_efficiency_per_epoch']:.2%}")
        print(f"   💡 Recommendation: {efficiency_info['recommendation']}")


class BatchTrioExtractor:
    """
    Helper class to extract multiple trios from a batch of images efficiently.
    """
    
    def __init__(self, patch_size: int = 64, step_size: int = 64):
        self.patch_size = patch_size
        self.step_size = step_size
    
    def extract_all_trios_from_image(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract ALL possible context trios from an image.
        
        Args:
            image: Preprocessed image tensor [C, H, W]
            
        Returns:
            List of (context_trio, target_patch) tuples
        """
        C, H, W = image.shape
        
        # Calculate patches grid based on step_size
        patches_H = (H - self.patch_size) // self.step_size + 1
        patches_W = (W - self.patch_size) // self.step_size + 1
        
        if patches_H < 2 or patches_W < 2:
            return []
        
        trios = []
        # Extract all possible 2x2 blocks
        for row in range(patches_H - 1):
            for col in range(patches_W - 1):
                # Calculate actual pixel positions based on step_size
                top_y = row * self.step_size
                left_x = col * self.step_size
                
                # Extract 2x2 block of patches
                top_left = image[:, top_y:top_y + self.patch_size, left_x:left_x + self.patch_size]
                top_right = image[:, top_y:top_y + self.patch_size, left_x + self.step_size:left_x + self.step_size + self.patch_size]
                bottom_left = image[:, top_y + self.step_size:top_y + self.step_size + self.patch_size, left_x:left_x + self.patch_size]
                bottom_right = image[:, top_y + self.step_size:top_y + self.step_size + self.patch_size, left_x + self.step_size:left_x + self.step_size + self.patch_size]
                
                # Create context trio: [TL, TR, BL]
                context_trio = torch.cat([top_left, top_right, bottom_left], dim=0)
                trios.append((context_trio, bottom_right))
        
        return trios
    
    def extract_random_trios_from_images(
        self, 
        images: List[torch.Tensor], 
        trios_per_image: int = 10
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract random trios from multiple images.
        
        Args:
            images: List of preprocessed image tensors
            trios_per_image: Number of trios to extract per image
            
        Returns:
            List of (context_trio, target_patch) tuples
        """
        all_trios = []
        
        for image in images:
            # Get all possible trios from this image
            image_trios = self.extract_all_trios_from_image(image)
            
            if image_trios:
                # Randomly sample from available trios
                num_samples = min(trios_per_image, len(image_trios))
                selected_trios = random.sample(image_trios, num_samples)
                all_trios.extend(selected_trios)
        
        return all_trios


class EvaluationDataset:
    """
    Simple evaluation dataset that loads specific images for consistent evaluation.
    """
    
    def __init__(
        self,
        dataset_path: str,
        patch_size: int = 64,  # DEPRECATED: Use trio_block_size instead
        trio_block_size: int = None,
        step_size: int = 64,
        padding_pixels: int = 64,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        enable_edge_sampling: bool = True,
        num_eval_images: int = 10,
        skip_images: int = 0,
        lmdb_path: str = None
    ):
        """Initialize evaluation dataset.

        Args:
            skip_images: Number of images to skip (for selecting unseen images after training set)
            lmdb_path: Path to LMDB cache (if available)
        """
        if trio_block_size is None:
            trio_block_size = patch_size  # Backward compatibility

        self.trio_block_size = trio_block_size
        self.patch_size = patch_size
        self.step_size = step_size
        self.trio_extractor = BatchTrioExtractor(trio_block_size, step_size)
        
        # Initialize image loader
        self.image_loader = ImageLoader(
            dataset_path=dataset_path,
            patch_size=trio_block_size,
            step_size=step_size,
            padding_pixels=padding_pixels,
            normalize_range=normalize_range,
            enable_edge_sampling=enable_edge_sampling,
            max_images=num_eval_images,
            skip_images=skip_images,
            lmdb_path=lmdb_path,
        )

        label = "unseen" if skip_images > 0 else "train"
        print(f"📊 EvaluationDataset ({label}): {len(self.image_loader)} images for evaluation")
        print(f"   🔲 Patch size: {trio_block_size}x{trio_block_size}")
    
    def get_eval_samples(self, max_samples_per_image: int = 50) -> List[Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]]]:
        """
        Get evaluation samples with positions for consistent evaluation.
        
        Returns:
            List of (image, samples) where samples contain (context_trio, target_patch, position)
        """
        eval_samples = []
        
        for img_idx in range(len(self.image_loader)):
            try:
                # Load image
                image = self.image_loader[img_idx]
                
                # Extract all trios with positions
                trios_with_positions = self._extract_trios_with_positions(image)
                
                # Limit number of samples
                if len(trios_with_positions) > max_samples_per_image:
                    trios_with_positions = random.sample(trios_with_positions, max_samples_per_image)
                
                eval_samples.append((image, trios_with_positions))
                
            except Exception as e:
                print(f"⚠️ Failed to load eval image {img_idx}: {e}")
                continue
        
        return eval_samples
    
    def _extract_trios_with_positions(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
        """Extract trios with their positions for evaluation using flexible patch system."""
        C, H, W = image.shape
        
        # Calculate trio blocks grid based on trio_block_size
        trio_blocks_H = H // self.trio_block_size - 1  # -1 because we need 2x2 blocks
        trio_blocks_W = W // self.trio_block_size - 1
        
        if trio_blocks_H <= 0 or trio_blocks_W <= 0:
            return []
        
        trios_with_positions = []
        # Extract all possible 2x2 blocks with positions using flexible system
        for row in range(trio_blocks_H):
            for col in range(trio_blocks_W):
                try:
                    context_trio, target_patch = extract_flexible_context_trio(
                        image, self.trio_block_size, row, col
                    )
                    
                    # Position is the grid position where the target patch belongs
                    position = (row, col)
                    
                    trios_with_positions.append((context_trio, target_patch, position))
                except Exception as e:
                    # Skip this trio if extraction fails
                    continue
        
        return trios_with_positions
    
    def get_image(self, idx: int) -> torch.Tensor:
        """Get a specific image by index."""
        return self.image_loader[idx]
    
    def __len__(self) -> int:
        """Return number of evaluation images."""
        return len(self.image_loader) 
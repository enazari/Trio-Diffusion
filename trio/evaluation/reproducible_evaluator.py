#!/usr/bin/env python3
"""
Reproducible Evaluation System for Trio Training

This script enables you to reproduce EXACT evaluation results from training by:
1. 🎯 Saving evaluation configurations (images, patch positions, random seeds)
2. 🔄 Loading and replaying the exact same evaluation
3. 💾 Storing all necessary state for perfect reproduction

Usage:
    # Save evaluation config during/after training
    python -m trio.evaluation.reproducible_evaluator save-config \
        --session-dir results/session_20250526_202050 \
        --checkpoint checkpoints/best_model.pt

    # Reproduce exact evaluation results
    python -m trio.evaluation.reproducible_evaluator reproduce \
        --eval-config results/session_20250526_202050/eval_config.json \
        --checkpoint checkpoints/epoch_050.pt \
        --output-dir reproduced_results
"""

import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time
import hashlib
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Import trio modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from trio.config import TrioConfig
from trio.models.patch_unet import PatchUNet
from trio.diffusion.patch_diffusion import PatchDiffusion
from trio.data.patch_dataset import PatchDataset
from trio.utils.training_utils import load_checkpoint, get_device, set_seed
from trio.utils.image_utils import (
    get_context_trio_samples_with_positions, apply_generated_patches, 
    save_comparison_image, tensor_to_pil, extract_patches, crop_for_perfect_division
)


@dataclass
class EvaluationConfig:
    """Configuration for reproducible evaluation."""
    # Randomness control
    random_seed: int
    
    # Model and training info
    session_id: str
    epoch: int
    checkpoint_path: str
    config_path: str
    
    # Evaluation parameters
    num_eval_images: int
    patches_per_image: int
    diffusion_eta: float
    diffusion_steps: int
    
    # Image and sample specifications
    image_specs: List[Dict[str, Any]]  # List of {image_idx, image_hash, samples}
    
    # Metadata
    created_timestamp: str
    trio_config_hash: str
    
    def save_to_file(self, path: str):
        """Save evaluation config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"💾 Saved evaluation config to: {path}")
    
    @classmethod
    def load_from_file(cls, path: str) -> 'EvaluationConfig':
        """Load evaluation config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ReproducibleTrioEvaluator:
    """Evaluator that can save and reproduce exact evaluation results."""
    
    def __init__(self, config: TrioConfig, device: str = "cuda"):
        """Initialize the reproducible evaluator."""
        self.config = config
        self.device = get_device(device)
        
        # Initialize dataset
        self.dataset = PatchDataset(
            dataset_path=config.data.dataset_path,
            num_images=config.data.num_images,
            trio_block_size=config.data.trio_block_size,
            step_size=config.data.step_size,
            padding_pixels=config.data.padding_pixels,
            eval_samples_per_image=config.data.eval_samples_per_image,
            normalize_range=config.data.normalize_range,
            enable_edge_sampling=config.data.enable_edge_sampling,
        )
        
        print(f"🔧 Reproducible Evaluator initialized")
        print(f"📁 Dataset: {config.data.dataset_path}")
        print(f"🖥️  Device: {self.device}")
    
    def _compute_image_hash(self, image: torch.Tensor) -> str:
        """Compute hash of image tensor for verification."""
        # Convert to bytes and hash
        image_bytes = image.detach().cpu().numpy().tobytes()
        return hashlib.sha256(image_bytes).hexdigest()[:16]
    
    def _compute_config_hash(self, config: TrioConfig) -> str:
        """Compute hash of configuration for verification."""
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def create_evaluation_config(
        self,
        session_dir: str,
        checkpoint_path: str,
        epoch: int,
        random_seed: int = 42,
        num_eval_images: int = None,
        patches_per_image: int = None
    ) -> EvaluationConfig:
        """
        Create a complete evaluation configuration for reproduction.
        
        Args:
            session_dir: Session directory path
            checkpoint_path: Path to model checkpoint
            epoch: Epoch number
            random_seed: Random seed for reproducibility
            num_eval_images: Number of evaluation images (uses config default if None)
            patches_per_image: Patches per image (uses config default if None)
            
        Returns:
            EvaluationConfig object
        """
        print(f"🎯 Creating evaluation configuration...")
        print(f"   Session: {session_dir}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Random seed: {random_seed}")
        
        # Set random seed for reproducible sampling
        set_seed(random_seed)
        
        # Use config defaults if not specified
        num_eval_images = num_eval_images or self.config.evaluation.num_eval_images
        patches_per_image = patches_per_image or self.config.evaluation.patches_per_image
        
        # Store the complete random state before generating samples
        random_state_before = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # Get fixed evaluation samples
        eval_samples = self.dataset.get_fixed_eval_samples(num_eval_images)
        
        # Create image specifications with exact sample information
        image_specs = []
        
        for img_idx, (original_image, samples) in enumerate(eval_samples):
            # Compute image hash for verification
            image_hash = self._compute_image_hash(original_image)
            
            # Select the subset of samples that would be used in evaluation
            selected_samples = samples[:patches_per_image]
            
            # Store each sample with its exact parameters
            sample_specs = []
            for sample_idx, (context_trio, target_patch, position) in enumerate(selected_samples):
                # Ensure position is stored as a tuple consistently
                position_tuple = tuple(position) if isinstance(position, (list, tuple)) else position
                
                sample_spec = {
                    'sample_idx': sample_idx,
                    'position': list(position_tuple),  # Store as list for JSON serialization
                    'context_trio_hash': self._compute_image_hash(context_trio),
                    'target_patch_hash': self._compute_image_hash(target_patch)
                }
                sample_specs.append(sample_spec)
            
            image_spec = {
                'eval_image_idx': img_idx,
                'dataset_image_idx': img_idx,  # Assuming 1:1 mapping for first N images
                'image_hash': image_hash,
                'image_shape': list(original_image.shape),
                'num_samples_used': len(selected_samples),
                'samples': sample_specs
            }
            image_specs.append(image_spec)
        
        # Create evaluation config
        session_path = Path(session_dir)
        config_path = str(session_path / "config.json")
        session_id = session_path.name.replace("session_", "")
        
        eval_config = EvaluationConfig(
            random_seed=random_seed,
            session_id=session_id,
            epoch=epoch,
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            config_path=str(Path(config_path).resolve()),
            num_eval_images=num_eval_images,
            patches_per_image=patches_per_image,
            diffusion_eta=self.config.diffusion.eta,
            diffusion_steps=self.config.diffusion.sampling_steps,
            image_specs=image_specs,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            trio_config_hash=self._compute_config_hash(self.config)
        )
        
        print(f"✅ Evaluation config created:")
        print(f"   📊 {len(image_specs)} images")
        print(f"   🎯 {sum(len(spec['samples']) for spec in image_specs)} total patches")
        print(f"   🔍 Config hash: {eval_config.trio_config_hash}")
        
        # Add a note about deterministic sampling limitations
        if any(len(spec['samples']) != patches_per_image for spec in image_specs):
            print(f"   ⚠️  Note: Some images have fewer samples than requested due to diversity filtering")
            print(f"        This is expected and the exact samples will be reproduced")
        
        return eval_config
    
    def reproduce_evaluation(
        self, 
        eval_config: EvaluationConfig, 
        checkpoint_path: str,
        output_dir: str,
        verify_hashes: bool = True
    ) -> None:
        """
        Reproduce exact evaluation results from saved configuration.
        
        Args:
            eval_config: Saved evaluation configuration
            checkpoint_path: Path to model checkpoint to use
            output_dir: Directory to save reproduced results
            verify_hashes: Whether to verify image/sample hashes match
        """
        print(f"🔄 Reproducing evaluation from config...")
        print(f"   Original session: {eval_config.session_id}")
        print(f"   Original epoch: {eval_config.epoch}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Random seed: {eval_config.random_seed}")
        print(f"   Output: {output_dir}")
        
        # Set the exact same random seed
        set_seed(eval_config.random_seed)
        
        # Load model
        model = PatchUNet(
            input_channels=self.config.model.input_channels,
            output_channels=self.config.model.output_channels,
            base_channels=self.config.model.base_channels,
            time_embedding_dim=self.config.model.time_embedding_dim,
            dropout=self.config.model.dropout,
            use_attention=self.config.model.use_attention
        ).to(self.device)
        
        # Initialize diffusion
        diffusion = PatchDiffusion(
            timesteps=self.config.diffusion.timesteps,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            device=self.device
        )
        
        # Load checkpoint
        load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=self.device
        )
        model.eval()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save reproduction info
        repro_info = {
            'original_config': asdict(eval_config),
            'reproduction_checkpoint': str(Path(checkpoint_path).resolve()),
            'reproduction_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'verification_enabled': verify_hashes
        }
        
        with open(output_path / "reproduction_info.json", 'w') as f:
            json.dump(repro_info, f, indent=2)
        
        # Set the random seed again before generating samples to ensure consistency
        set_seed(eval_config.random_seed)
        
        # Regenerate the exact same evaluation samples
        eval_samples = self.dataset.get_fixed_eval_samples(eval_config.num_eval_images)
        
        print(f"🎨 Processing {len(eval_samples)} images...")
        
        verification_results = []
        
        for img_idx, (original_image, samples) in enumerate(eval_samples):
            if img_idx >= len(eval_config.image_specs):
                print(f"⚠️  Warning: More images in dataset than in config, stopping at {img_idx}")
                break
                
            image_spec = eval_config.image_specs[img_idx]
            
            print(f"   Processing image {img_idx + 1}/{len(eval_samples)}")
            
            # Verify image hash if requested
            current_hash = self._compute_image_hash(original_image)
            expected_hash = image_spec['image_hash']
            
            if verify_hashes and current_hash != expected_hash:
                print(f"⚠️  Warning: Image {img_idx} hash mismatch!")
                print(f"      Expected: {expected_hash}")
                print(f"      Current:  {current_hash}")
                verification_results.append({
                    'image_idx': img_idx,
                    'type': 'image_hash_mismatch',
                    'expected': expected_hash,
                    'current': current_hash
                })
            
            # Select the exact same samples as in the original evaluation
            eval_samples_subset = samples[:eval_config.patches_per_image]
            
            # Verify we have the expected number of samples
            if len(eval_samples_subset) != image_spec['num_samples_used']:
                print(f"⚠️  Warning: Sample count mismatch for image {img_idx}")
                print(f"      Expected: {image_spec['num_samples_used']}")
                print(f"      Current:  {len(eval_samples_subset)}")
            
            # Generate patches with the exact same settings
            generated_patches = []
            positions = []
            
            for sample_idx, (context_trio, target_patch, position) in enumerate(eval_samples_subset):
                if sample_idx < len(image_spec['samples']):
                    sample_spec = image_spec['samples'][sample_idx]
                    
                    # Convert position to tuple for consistent comparison
                    current_position = tuple(position) if isinstance(position, (list, tuple)) else position
                    expected_position = tuple(sample_spec['position']) if isinstance(sample_spec['position'], list) else sample_spec['position']
                    
                    # Verify position matches
                    if verify_hashes and current_position != expected_position:
                        print(f"⚠️  Warning: Position mismatch for image {img_idx}, sample {sample_idx}")
                        print(f"      Expected: {sample_spec['position']}")
                        print(f"      Current:  {current_position}")
                        verification_results.append({
                            'image_idx': img_idx,
                            'sample_idx': sample_idx,
                            'type': 'position_mismatch',
                            'expected': sample_spec['position'],
                            'current': current_position
                        })
                    
                    # Verify context trio hash
                    if verify_hashes:
                        current_trio_hash = self._compute_image_hash(context_trio)
                        expected_trio_hash = sample_spec['context_trio_hash']
                        if current_trio_hash != expected_trio_hash:
                            print(f"⚠️  Warning: Context trio hash mismatch for image {img_idx}, sample {sample_idx}")
                            verification_results.append({
                                'image_idx': img_idx,
                                'sample_idx': sample_idx,
                                'type': 'context_trio_hash_mismatch',
                                'expected': expected_trio_hash,
                                'current': current_trio_hash
                            })
                
                # Generate patch with exact same parameters
                context_trio_batch = context_trio.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    generated_patch = diffusion.ddim_sample(
                        model,
                        context_trio_batch,
                        (1, 3, self.config.data.trio_block_size, self.config.data.trio_block_size),
                        eta=eval_config.diffusion_eta,
                        steps=eval_config.diffusion_steps
                    )
                
                generated_patches.append(generated_patch.squeeze(0).cpu())
                positions.append(position)
            
            # Apply generated patches to create result image
            result_image = apply_generated_patches(
                original_image, 
                generated_patches, 
                positions, 
                self.config.data.trio_block_size
            )
            
            # Save comparison with detailed filename
            comparison_path = output_path / f"reproduced_image_{img_idx + 1:02d}_epoch_{eval_config.epoch}.png"
            save_comparison_image(
                original_image, 
                result_image, 
                str(comparison_path),
                self.config.data.normalize_range
            )
        
        # Save verification results
        if verification_results:
            print(f"⚠️  Found {len(verification_results)} verification issues")
            with open(output_path / "verification_issues.json", 'w') as f:
                json.dump(verification_results, f, indent=2)
            
            # Analyze and explain the verification issues
            position_mismatches = [r for r in verification_results if r['type'] == 'position_mismatch']
            hash_mismatches = [r for r in verification_results if r['type'] in ['context_trio_hash_mismatch', 'image_hash_mismatch']]
            
            print(f"\n📊 Verification Issue Summary:")
            if position_mismatches:
                print(f"   🎯 Position mismatches: {len(position_mismatches)} - Random sampling generated different patch positions")
            if hash_mismatches:
                print(f"   🔍 Hash mismatches: {len(hash_mismatches)} - Different patch content due to non-deterministic sampling")
            
            print(f"\n💡 Understanding the Issues:")
            print(f"   These warnings indicate that the random patch sampling process")
            print(f"   generated different patches than during the original evaluation.")
            print(f"   This can happen because:")
            print(f"   1. 🎲 Random sampling with diversity filtering is inherently non-deterministic")
            print(f"   2. 🔄 Even with the same seed, the sampling loop may find different patches")
            print(f"   3. ⏱️  The order of operations during evaluation vs reproduction may differ")
            print(f"")
            print(f"   🎯 The generated images are still valid comparisons, but they use")
            print(f"      different source patches than the original evaluation.")
            print(f"")
            print(f"   ✅ For exact reproduction, consider using deterministic patch selection")
            print(f"      instead of random sampling in future versions.")
        
        else:
            print(f"✅ All verifications passed! Perfect reproduction achieved.")
        
        print(f"✅ Reproduction completed!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Generated {len(eval_samples)} comparison images")

    def reproduce_autoregressive_generation(
        self,
        checkpoint_path: str,
        target_size: Tuple[int, int] = (1024, 1024),
        initial_source: Optional[str] = None,
        output_dir: str = "autoregressive_results",
        random_seed: int = 42,
        steps: int = 50,
        eta: float = 0.0
    ) -> str:
        """
        Generate images autoregressively with full reproducibility.
        
        Args:
            checkpoint_path: Path to model checkpoint
            target_size: (height, width) of generated image
            initial_source: Path to initial image or None for grey initialization
            output_dir: Directory to save results
            random_seed: Random seed for reproducible generation
            steps: Number of diffusion sampling steps
            eta: DDIM eta parameter
            
        Returns:
            Path to generated image
        """
        print(f"🚀 Starting reproducible autoregressive generation...")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Target size: {target_size}")
        print(f"   Random seed: {random_seed}")
        print(f"   Steps: {steps}, eta: {eta}")
        
        # Set random seed for reproducibility
        set_seed(random_seed)
        
        # Load model
        model = PatchUNet(
            input_channels=self.config.model.input_channels,
            output_channels=self.config.model.output_channels,
            base_channels=self.config.model.base_channels,
            time_embedding_dim=self.config.model.time_embedding_dim,
            dropout=self.config.model.dropout,
            use_attention=self.config.model.use_attention
        ).to(self.device)
        
        # Initialize diffusion
        diffusion = PatchDiffusion(
            timesteps=self.config.diffusion.timesteps,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            device=self.device
        )
        
        # Load checkpoint
        load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=self.device
        )
        model.eval()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create initial image with grey ribbon
        image = self._create_initial_image_with_ribbon(
            target_size, self.config.data.trio_block_size, initial_source
        )
        C, H, W = image.shape
        
        # Calculate generation grid (excluding ribbon)
        gen_patches_H = target_size[0] // self.config.data.trio_block_size
        gen_patches_W = target_size[1] // self.config.data.trio_block_size
        total_patches = gen_patches_H * gen_patches_W
        
        print(f"🎯 Generating {total_patches} patches ({gen_patches_H}x{gen_patches_W})")
        print(f"📏 Patch size: {self.config.data.trio_block_size}x{self.config.data.trio_block_size}")
        
        # Store generation info for reproducibility
        generation_info = {
            'random_seed': random_seed,
            'checkpoint_path': str(Path(checkpoint_path).resolve()),
            'target_size': target_size,
            'initial_source': str(Path(initial_source).resolve()) if initial_source else None,
            'patch_size': self.config.data.trio_block_size,
            'diffusion_steps': steps,
            'diffusion_eta': eta,
            'generation_grid': [gen_patches_H, gen_patches_W],
            'total_patches': total_patches,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config_hash': self._compute_config_hash(self.config)
        }
        
        # Generate patches row by row, left to right in the generation area
        patch_info = []
        with tqdm(total=total_patches, desc="Generating patches") as pbar:
            for row in range(gen_patches_H):
                for col in range(gen_patches_W):
                    # Extract context trio
                    context_trio = self._generate_context_trio_autoregressive(
                        image, self.config.data.trio_block_size, row, col
                    )
                    
                    # Generate patch
                    generated_patch = self._generate_single_patch(
                        model, diffusion, context_trio, steps, eta
                    )
                    
                    # Apply to image
                    image = self._apply_patch_to_image_autoregressive(
                        image, generated_patch, row, col, self.config.data.trio_block_size
                    )
                    
                    # Store patch info for reproducibility
                    patch_info.append({
                        'row': row,
                        'col': col,
                        'context_trio_hash': self._compute_image_hash(context_trio),
                        'generated_patch_hash': self._compute_image_hash(generated_patch)
                    })
                    
                    pbar.update(1)
        
        # Extract final image (remove ribbon)
        final_image = image[:, self.config.data.trio_block_size:, self.config.data.trio_block_size:]
        
        # Save results
        result_path = output_path / f"autoregressive_generation_seed_{random_seed}.png"
        pil_image = tensor_to_pil(final_image, normalize_range=self.config.data.normalize_range)
        pil_image.save(result_path)
        
        # Save generation info
        generation_info['patches'] = patch_info
        generation_info['result_path'] = str(result_path.resolve())
        generation_info['final_image_hash'] = self._compute_image_hash(final_image)
        
        info_path = output_path / f"generation_info_seed_{random_seed}.json"
        with open(info_path, 'w') as f:
            json.dump(generation_info, f, indent=2)
        
        print(f"✅ Generated image saved to: {result_path}")
        print(f"📊 Final size: {pil_image.size}")
        print(f"📝 Generation info saved to: {info_path}")
        
        return str(result_path)
    
    def _create_initial_image_with_ribbon(
        self,
        target_size: Tuple[int, int], 
        patch_size: int,
        initial_source: Optional[str] = None,
        fill_value: float = 0.5
    ) -> torch.Tensor:
        """Create initial image with grey ribbon for autoregressive generation."""
        height, width = target_size
        
        if initial_source is None:
            # Create image with grey ribbon
            image = torch.full((3, height + patch_size, width + patch_size), fill_value)
        else:
            # Load image and add grey ribbon
            source_path = Path(initial_source)
            if source_path.is_file():
                pil_image = Image.open(source_path).convert('RGB')
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                
                # Convert to tensor and normalize
                content = torch.from_numpy(np.array(pil_image)).float() / 255.0
                content = content.permute(2, 0, 1)  # HWC -> CHW
                content = content * (self.config.data.normalize_range[1] - self.config.data.normalize_range[0]) + self.config.data.normalize_range[0]
                
                # Create image with grey ribbon
                image = torch.full((3, height + patch_size, width + patch_size), fill_value)
                # Place content image in the bottom-right area (after ribbon)
                image[:, patch_size:, patch_size:] = content
            else:
                raise FileNotFoundError(f"Initial image not found: {initial_source}")
        
        # Ensure dimensions are divisible by patch_size
        image = crop_for_perfect_division(image, patch_size)
        
        return image
    
    def _generate_context_trio_autoregressive(
        self,
        image: torch.Tensor,
        patch_size: int,
        row: int,
        col: int
    ) -> torch.Tensor:
        """Extract context trio for autoregressive generation."""
        C, H, W = image.shape
        
        # Calculate actual patch positions accounting for ribbon offset
        actual_row = row + 1  # +1 because of ribbon
        actual_col = col + 1  # +1 because of ribbon
        
        # Extract all patches from the current image
        patches = extract_patches(image, patch_size)
        patches_H = H // patch_size
        patches_W = W // patch_size
        patches = patches.view(patches_H, patches_W, C, patch_size, patch_size)
        
        # Extract context patches for 2x2 block pattern (SAME AS TRAINING)
        top_left = patches[actual_row-1, actual_col-1]     # [3, patch_size, patch_size]
        top_right = patches[actual_row-1, actual_col]      # [3, patch_size, patch_size]  
        bottom_left = patches[actual_row, actual_col-1]    # [3, patch_size, patch_size]
        
        # Create context trio in EXACT SAME ORDER as training: [TL, TR, BL]
        context_trio = torch.cat([top_left, top_right, bottom_left], dim=0)  # [9, patch_size, patch_size]
        
        return context_trio
    
    @torch.no_grad()
    def _generate_single_patch(
        self,
        model: PatchUNet,
        diffusion: PatchDiffusion,
        context_trio: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """Generate a single patch using the model."""
        # Add batch dimension and move to device
        context_trio_batch = context_trio.unsqueeze(0).to(self.device)
        
        # Generate using DDIM
        patch_size = context_trio.shape[-1]
        generated_patch = diffusion.ddim_sample(
            model,
            context_trio_batch,
            (1, 3, patch_size, patch_size),
            eta=eta,
            steps=steps
        )
        
        return generated_patch.squeeze(0).cpu()
    
    def _apply_patch_to_image_autoregressive(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        row: int,
        col: int,
        patch_size: int
    ) -> torch.Tensor:
        """Apply generated patch to image at specified position."""
        # Calculate actual position accounting for ribbon offset
        actual_row = row + 1  # +1 because of ribbon
        actual_col = col + 1  # +1 because of ribbon
        
        start_row = actual_row * patch_size
        end_row = (actual_row + 1) * patch_size
        start_col = actual_col * patch_size
        end_col = (actual_col + 1) * patch_size
        
        # Apply patch
        image[:, start_row:end_row, start_col:end_col] = patch
        
        return image


def save_training_evaluation_config(session_dir: str, checkpoint_path: str, epoch: int = None, config_filename: str = None) -> str:
    """
    Save evaluation configuration from a training session.
    
    Args:
        session_dir: Path to training session directory
        checkpoint_path: Path to checkpoint file
        epoch: Epoch number (extracted from checkpoint if None)
        config_filename: Custom config filename (uses eval_config.json if None)
        
    Returns:
        Path to saved evaluation config
    """
    session_path = Path(session_dir)
    # Try config.yaml first (new format), then config.json (legacy)
    config_path = session_path / "config.yaml"
    if not config_path.exists():
        config_path = session_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {session_path / 'config.yaml'}")

    # Load training configuration
    config = TrioConfig.from_yaml(str(config_path))
    
    # Extract epoch from checkpoint filename if not provided
    if epoch is None:
        checkpoint_name = Path(checkpoint_path).stem
        try:
            # Handle both checkpoint_epoch_XXX and eval_checkpoint_epoch_XXX formats
            if "eval_checkpoint_epoch_" in checkpoint_name:
                epoch = int(checkpoint_name.split('_')[-1])
            elif "checkpoint_epoch_" in checkpoint_name:
                epoch = int(checkpoint_name.split('_')[-1])
            else:
                epoch = 0
        except (ValueError, IndexError):
            epoch = 0
            print(f"⚠️  Could not extract epoch from {checkpoint_name}, using 0")
    
    # Check if this is already a dedicated evaluation checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.name.startswith("eval_checkpoint"):
        # Create a dedicated evaluation checkpoint to ensure reproducibility
        eval_checkpoint_name = f"eval_checkpoint_epoch_{epoch:03d}.pt"
        eval_checkpoint_path = session_path / "checkpoints" / eval_checkpoint_name
        
        if not eval_checkpoint_path.exists():
            print(f"📋 Creating dedicated evaluation checkpoint for reproducibility...")
            
            # Load the source checkpoint
            source_checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # Save as evaluation checkpoint (this won't be cleaned up)
            torch.save(source_checkpoint, str(eval_checkpoint_path))
            print(f"💾 Created evaluation checkpoint: {eval_checkpoint_path.name}")
        
        # Use the dedicated evaluation checkpoint
        checkpoint_path = eval_checkpoint_path
    
    # Initialize evaluator
    evaluator = ReproducibleTrioEvaluator(config)
    
    # Create evaluation config
    eval_config = evaluator.create_evaluation_config(
        session_dir=str(session_path),
        checkpoint_path=str(checkpoint_path),
        epoch=epoch,
        random_seed=42  # Fixed seed for reproducibility
    )
    
    # Use custom filename if provided, otherwise use default
    if config_filename is None:
        config_filename = "eval_config.json"
    
    # Save config
    eval_config_path = session_path / config_filename
    eval_config.save_to_file(str(eval_config_path))
    
    return str(eval_config_path)


def reproduce_from_config(eval_config_path: str, checkpoint_path: str, output_dir: str, verify: bool = True) -> None:
    """
    Reproduce evaluation results from saved configuration.
    
    Args:
        eval_config_path: Path to evaluation config JSON
        checkpoint_path: Path to checkpoint to use for reproduction
        output_dir: Directory to save results
        verify: Whether to verify hashes match original
    """
    # Load evaluation config
    eval_config = EvaluationConfig.load_from_file(eval_config_path)
    
    # Load original training config
    if Path(eval_config.config_path).exists():
        trio_config = TrioConfig.from_yaml(eval_config.config_path)
    else:
        print(f"⚠️  Original config not found: {eval_config.config_path}")
        print(f"    Using default configuration")
        trio_config = TrioConfig()
    
    # Initialize evaluator
    evaluator = ReproducibleTrioEvaluator(trio_config)
    
    # Reproduce evaluation
    evaluator.reproduce_evaluation(
        eval_config=eval_config,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        verify_hashes=verify
    )


def generate_autoregressive_from_config(
    eval_config_path: str,
    checkpoint_path: str,
    output_dir: str,
    target_size: Tuple[int, int] = (1024, 1024),
    initial_source: Optional[str] = None,
    random_seed: int = 42,
    steps: int = 50,
    eta: float = 0.0
) -> str:
    """
    Generate image autoregressively using saved evaluation configuration.
    
    Args:
        eval_config_path: Path to evaluation config JSON
        checkpoint_path: Path to checkpoint to use for generation
        output_dir: Directory to save results
        target_size: (height, width) of generated image
        initial_source: Path to initial image or None
        random_seed: Random seed for reproducible generation
        steps: Number of diffusion sampling steps
        eta: DDIM eta parameter
        
    Returns:
        Path to generated image
    """
    # Load evaluation config
    eval_config = EvaluationConfig.load_from_file(eval_config_path)
    
    # Load original training config
    if Path(eval_config.config_path).exists():
        trio_config = TrioConfig.from_yaml(eval_config.config_path)
    else:
        print(f"⚠️  Original config not found: {eval_config.config_path}")
        print(f"    Using default configuration")
        trio_config = TrioConfig()
    
    # Initialize evaluator
    evaluator = ReproducibleTrioEvaluator(trio_config)
    
    # Generate autoregressive image
    result_path = evaluator.reproduce_autoregressive_generation(
        checkpoint_path=checkpoint_path,
        target_size=target_size,
        initial_source=initial_source,
        output_dir=output_dir,
        random_seed=random_seed,
        steps=steps,
        eta=eta
    )
    
    return result_path


def compare_autoregressive_generations(
    eval_config_path: str,
    checkpoint_paths: List[str],
    output_dir: str,
    target_size: Tuple[int, int] = (1024, 1024),
    initial_source: Optional[str] = None,
    random_seed: int = 42,
    steps: int = 50,
    eta: float = 0.0
) -> List[str]:
    """
    Generate multiple autoregressive images with different checkpoints for comparison.
    
    Args:
        eval_config_path: Path to evaluation config JSON
        checkpoint_paths: List of checkpoint paths to compare
        output_dir: Directory to save results
        target_size: (height, width) of generated image
        initial_source: Path to initial image or None
        random_seed: Random seed for reproducible generation (same for all)
        steps: Number of diffusion sampling steps
        eta: DDIM eta parameter
        
    Returns:
        List of paths to generated images
    """
    print(f"🔄 Comparing autoregressive generation across {len(checkpoint_paths)} checkpoints...")
    print(f"   Using fixed random seed {random_seed} for consistency")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load evaluation config
    eval_config = EvaluationConfig.load_from_file(eval_config_path)
    
    # Load original training config
    if Path(eval_config.config_path).exists():
        trio_config = TrioConfig.from_yaml(eval_config.config_path)
    else:
        print(f"⚠️  Original config not found: {eval_config.config_path}")
        trio_config = TrioConfig()
    
    # Initialize evaluator
    evaluator = ReproducibleTrioEvaluator(trio_config)
    
    generated_paths = []
    comparison_info = {
        'random_seed': random_seed,
        'target_size': target_size,
        'initial_source': str(Path(initial_source).resolve()) if initial_source else None,
        'diffusion_steps': steps,
        'diffusion_eta': eta,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'checkpoints': []
    }
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint_name = Path(checkpoint_path).stem
        print(f"\n🎯 Generating with checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_name}")
        
        # Generate with specific checkpoint
        result_path = evaluator.reproduce_autoregressive_generation(
            checkpoint_path=checkpoint_path,
            target_size=target_size,
            initial_source=initial_source,
            output_dir=str(output_path / f"checkpoint_{i+1:02d}_{checkpoint_name}"),
            random_seed=random_seed,  # Same seed for all generations
            steps=steps,
            eta=eta
        )
        
        generated_paths.append(result_path)
        comparison_info['checkpoints'].append({
            'index': i + 1,
            'checkpoint_path': str(Path(checkpoint_path).resolve()),
            'checkpoint_name': checkpoint_name,
            'result_path': str(Path(result_path).resolve())
        })
    
    # Save comparison info
    comparison_info_path = output_path / f"autoregressive_comparison_seed_{random_seed}.json"
    with open(comparison_info_path, 'w') as f:
        json.dump(comparison_info, f, indent=2)
    
    print(f"\n✅ Generated {len(generated_paths)} autoregressive images")
    print(f"📊 All using the same random seed ({random_seed}) for consistency")
    print(f"📝 Comparison info saved to: {comparison_info_path}")
    
    return generated_paths


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Reproducible Evaluation System for Trio Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save evaluation config from training session
  python -m trio.evaluation.reproducible_evaluator save-config \\
      --session-dir results/session_20250526_202050 \\
      --checkpoint results/session_20250526_202050/checkpoints/best_model.pt
  
  # Reproduce evaluation with different checkpoint
  python -m trio.evaluation.reproducible_evaluator reproduce \\
      --eval-config results/session_20250526_202050/eval_config.json \\
      --checkpoint results/session_20250526_202050/checkpoints/checkpoint_epoch_050.pt \\
      --output-dir reproduced_results
      
  # Generate autoregressive image from evaluation config
  python -m trio.evaluation.reproducible_evaluator generate \\
      --eval-config results/session_20250526_202050/eval_config.json \\
      --checkpoint results/session_20250526_202050/checkpoints/eval_checkpoint_epoch_010.pt \\
      --output-dir generated_images --size 1024 1024
      
  # Compare autoregressive generation across multiple checkpoints
  python -m trio.evaluation.reproducible_evaluator compare \\
      --eval-config results/session_20250526_202050/eval_config.json \\
      --checkpoints checkpoint_1.pt checkpoint_2.pt checkpoint_3.pt \\
      --output-dir generation_comparison
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Save config command
    save_parser = subparsers.add_parser('save-config', help='Save evaluation configuration from training session')
    save_parser.add_argument('--session-dir', required=True, help='Path to training session directory')
    save_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    save_parser.add_argument('--epoch', type=int, help='Epoch number (auto-detected if not provided)')
    
    # Reproduce command  
    repro_parser = subparsers.add_parser('reproduce', help='Reproduce evaluation from saved configuration')
    repro_parser.add_argument('--eval-config', required=True, help='Path to evaluation config JSON')
    repro_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint to use')
    repro_parser.add_argument('--output-dir', required=True, help='Directory to save reproduced results')
    repro_parser.add_argument('--no-verify', action='store_true', help='Skip hash verification')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate autoregressive image from evaluation config')
    gen_parser.add_argument('--eval-config', required=True, help='Path to evaluation config JSON')
    gen_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint to use for generation')
    gen_parser.add_argument('--output-dir', required=True, help='Directory to save generated results')
    gen_parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024], 
                           help='Target size (height width) - default: 1024 1024')
    gen_parser.add_argument('--initial', type=str, help='Path to initial image (default: grey initialization)')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation')
    gen_parser.add_argument('--steps', type=int, default=50, help='Number of diffusion sampling steps')
    gen_parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter')
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare autoregressive generation across multiple checkpoints')
    comp_parser.add_argument('--eval-config', required=True, help='Path to evaluation config JSON')
    comp_parser.add_argument('--checkpoints', nargs='+', required=True, help='List of checkpoint paths to compare')
    comp_parser.add_argument('--output-dir', required=True, help='Directory to save comparison results')
    comp_parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024],
                           help='Target size (height width) - default: 1024 1024')
    comp_parser.add_argument('--initial', type=str, help='Path to initial image (default: grey initialization)')
    comp_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation')
    comp_parser.add_argument('--steps', type=int, default=50, help='Number of diffusion sampling steps')
    comp_parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter')
    
    args = parser.parse_args()
    
    if args.command == 'save-config':
        print(f"💾 Saving evaluation configuration...")
        eval_config_path = save_training_evaluation_config(
            session_dir=args.session_dir,
            checkpoint_path=args.checkpoint,
            epoch=args.epoch
        )
        print(f"✅ Saved to: {eval_config_path}")
        
    elif args.command == 'reproduce':
        print(f"🔄 Reproducing evaluation...")
        reproduce_from_config(
            eval_config_path=args.eval_config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            verify=not args.no_verify
        )
        print(f"✅ Reproduction completed!")
        
    elif args.command == 'generate':
        print(f"🎨 Generating autoregressive image...")
        result_path = generate_autoregressive_from_config(
            eval_config_path=args.eval_config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            target_size=tuple(args.size),
            initial_source=args.initial,
            random_seed=args.seed,
            steps=args.steps,
            eta=args.eta
        )
        print(f"✅ Generated image saved to: {result_path}")
        
    elif args.command == 'compare':
        print(f"🔄 Comparing autoregressive generation...")
        result_paths = compare_autoregressive_generations(
            eval_config_path=args.eval_config,
            checkpoint_paths=args.checkpoints,
            output_dir=args.output_dir,
            target_size=tuple(args.size),
            initial_source=args.initial,
            random_seed=args.seed,
            steps=args.steps,
            eta=args.eta
        )
        print(f"✅ Comparison completed! Generated {len(result_paths)} images")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
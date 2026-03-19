"""Training utilities."""

import torch
import torch.optim as optim
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred_device: str = "cuda") -> torch.device:
    """Get the best available device."""
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_path: str,
    scaler=None,
    training_log: Optional[list] = None,
    ema_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Save checkpoint atomically to prevent corruption on compute clusters."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'ema_state_dict': ema_state_dict,
        'loss': loss,
        'config': config,
        'training_log': training_log[-10:] if training_log else [],
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'random_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    try:
        torch.save(checkpoint, temp)
        shutil.move(str(temp), str(target))
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: Optional[torch.device] = None,
    restore_random_state: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint and restore model/optimizer/scheduler states."""
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DataParallel/DDP state dict mismatch
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'module'):
        if not any(k.startswith('module.') for k in state_dict):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    else:
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if restore_random_state and 'random_state' in checkpoint:
        try:
            rs = checkpoint['random_state']
            if 'python' in rs:
                random.setstate(rs['python'])
            if 'numpy' in rs:
                np.random.set_state(rs['numpy'])
            if 'torch' in rs:
                torch.set_rng_state(rs['torch'])
            if rs.get('torch_cuda') and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs['torch_cuda'])
        except Exception as e:
            print(f"Warning: Could not restore random states: {e}")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'config': checkpoint.get('config', {}),
    }


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Cosine LR scheduler with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

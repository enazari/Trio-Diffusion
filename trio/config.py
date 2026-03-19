"""Configuration for the Trio pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import os
import shutil
import yaml


@dataclass
class DataConfig:
    dataset_path: str = ""
    num_images: int = 100
    trio_block_size: int = 64
    batch_size: int = 64
    step_size: int = 32
    samples_per_epoch: int = 50000
    num_workers: int = 8
    eval_samples_per_image: int = 250
    normalize_range: Tuple[float, float] = (-1.0, 1.0)
    enable_edge_sampling: bool = True

    @property
    def padding_pixels(self) -> int:
        return self.trio_block_size if self.enable_edge_sampling else 0


@dataclass
class ModelConfig:
    input_channels: int = 4
    output_channels: int = 3
    base_channels: int = 64
    time_embedding_dim: int = 128
    use_attention: bool = True
    dropout: float = 0.1
    use_position_encoding: bool = True
    use_coordinate_channels: bool = True


_BACKBONE_SPECS = {
    "clip": {"num_tokens": 49, "embedding_dim": 768},
    "dino": {"num_tokens": 256, "embedding_dim": 768},
    "none": {"num_tokens": 49, "embedding_dim": 768},
}


@dataclass
class BackboneConfig:
    name: str = "none"
    cross_attention_heads: int = 8
    use_token_position_encoding: bool = False

    @property
    def num_tokens(self) -> int:
        return _BACKBONE_SPECS.get(self.name, _BACKBONE_SPECS["none"])["num_tokens"]

    @property
    def embedding_dim(self) -> int:
        return _BACKBONE_SPECS.get(self.name, _BACKBONE_SPECS["none"])["embedding_dim"]


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "cosine"
    sampling_steps: int = 50
    eta: float = 0.0
    cfg_dropout_prob: float = 0.1
    guidance_scale: float = 2.0


@dataclass
class TrainingConfig:
    epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = False
    from_checkpoint: Optional[str] = None
    use_teacher_forcing: bool = False
    teacher_forcing_midpoint: float = 0.5
    teacher_forcing_temperature: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.9999
    scheduler_type: str = "cosine"
    cross_attention_gate_init: float = 0.5
    context_noise_prob: float = 0.3
    context_noise_scale: float = 0.2


@dataclass
class EvaluationConfig:
    checkpoint_every: int = 10
    num_eval_images: int = 10
    patches_per_image: int = 4
    output_dir: str = "evaluations"
    save_comparisons: bool = True
    generate_full_every: int = 25
    num_refinement_passes: int = 2
    generation_size: int = 256


@dataclass
class TrioConfig:
    data: DataConfig = None
    model: ModelConfig = None
    backbone: BackboneConfig = None
    diffusion: DiffusionConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None

    device: str = "cuda"
    seed: int = 42
    session: str = "experiment"

    session_id: str = None
    session_dir: str = None
    checkpoint_dir: str = None
    log_dir: str = None
    _config_yaml_path: str = None

    def __post_init__(self):
        if self.data is None: self.data = DataConfig()
        if self.model is None: self.model = ModelConfig()
        if self.backbone is None: self.backbone = BackboneConfig()
        if self.diffusion is None: self.diffusion = DiffusionConfig()
        if self.training is None: self.training = TrainingConfig()
        if self.evaluation is None: self.evaluation = EvaluationConfig()
        if self.session_dir is None:
            self._setup_session()

    def _setup_session(self):
        now = datetime.now()
        self.session_id = f"{self.session}_{now.strftime('%b').lower()}{now.day}-{now.strftime('%H:%M')}"
        self.session_dir = str(Path("sessions") / self.session_id)
        p = Path(self.session_dir)
        self.checkpoint_dir = str(p / "checkpoints")
        self.log_dir = str(p / "logs")
        self.evaluation.output_dir = str(p / "evaluations")

        for d in [self.checkpoint_dir, self.log_dir, self.evaluation.output_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        if self._config_yaml_path and Path(self._config_yaml_path).exists():
            shutil.copy2(self._config_yaml_path, p / "config.yaml")

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Session: {self.session_dir}")

    @classmethod
    def from_yaml(cls, filepath: str) -> 'TrioConfig':
        with open(filepath) as f:
            d = yaml.safe_load(f)

        config = cls.__new__(cls)
        config.data = DataConfig()
        config.model = ModelConfig()
        config.backbone = BackboneConfig()
        config.diffusion = DiffusionConfig()
        config.training = TrainingConfig()
        config.evaluation = EvaluationConfig()
        config.device = "cuda"
        config.seed = 42
        config.session = "experiment"
        config.session_id = None
        config.session_dir = None
        config.checkpoint_dir = None
        config.log_dir = None
        config._config_yaml_path = str(filepath)

        nested = {
            'data': config.data, 'model': config.model,
            'backbone': config.backbone, 'diffusion': config.diffusion,
            'training': config.training, 'evaluation': config.evaluation,
        }
        for key, value in d.items():
            if key in nested and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(nested[key], k):
                        # Skip read-only properties (e.g. num_tokens)
                        if isinstance(getattr(type(nested[key]), k, None), property):
                            continue
                        setattr(nested[key], k, v)
            elif hasattr(config, key):
                setattr(config, key, value)

        if config.session_dir is None:
            config._setup_session()
        return config


def load_config(config_name: str) -> TrioConfig:
    """Load config from configs/<config_name>.yaml, falling back to DATASET_PATH env var."""
    yaml_path = Path("configs") / f"{config_name}.yaml"
    if not yaml_path.exists():
        available = [f.stem for f in Path("configs").glob("*.yaml")] if Path("configs").exists() else []
        raise FileNotFoundError(f"Config '{config_name}' not found. Available: {', '.join(sorted(available))}")
    config = TrioConfig.from_yaml(str(yaml_path))
    if not config.data.dataset_path:
        config.data.dataset_path = os.environ.get("DATASET_PATH", "")
    return config

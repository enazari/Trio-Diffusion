"""
Autoregressive image generator for the Trio diffusion model (spatial inpainting).

Usage:
    python generate.py sessions/my_session/
    python generate.py sessions/my_session/ --size 1920 1080 --output out.png
    python generate.py sessions/my_session/ --fixed-global-context photo.jpg
        # backbone encodes photo as fixed global conditioning, generate at --size
    python generate.py sessions/my_session/ --fixed-global-context-fit photo.jpg
        # backbone encodes photo as fixed global conditioning, output matches photo dimensions
    python generate.py sessions/my_session/ --fixed-full-context photo.jpg
        # backbone encodes photo for global conditioning (if backbone exists) AND
        # actual photo patches are used as local TL/TR/BL context for each patch
    python generate.py sessions/my_session/ --text-seed "a sunset"
        # CLIP conditioned on text (requires CLIP backbone)
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Allow running from any directory (e.g. `python scripts/generate.py ...`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trio.config import TrioConfig
from trio.models.patch_unet import PatchUNet
from trio.diffusion.patch_diffusion import PatchDiffusion
from trio.backbones import build_backbone
from trio.backbones.clip import CLIPBackbone
from trio.diffusion.patch_diffusion import autoregressive_generate


def load_session(session_dir: str, device: torch.device):
    """Load model (EMA weights preferred), diffusion, backbone, and config."""
    p = Path(session_dir)
    config = TrioConfig.from_yaml(str(p / "config.yaml"))

    backbone = build_backbone(config)
    if backbone is not None:
        backbone = backbone.to(device).eval()

    use_gc = backbone is not None
    model = PatchUNet(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        base_channels=config.model.base_channels,
        time_embedding_dim=config.model.time_embedding_dim,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention,
        use_global_conditioning=use_gc,
        global_embedding_dim=config.backbone.embedding_dim,
        global_cross_attention_heads=config.backbone.cross_attention_heads,
        global_num_tokens=config.backbone.num_tokens if use_gc else 49,
        use_position_encoding=getattr(config.model, 'use_position_encoding', False),
        use_coordinate_channels=getattr(config.model, 'use_coordinate_channels', False),
        gate_init=getattr(config.training, 'cross_attention_gate_init', 0.0),
    ).to(device)

    ckpt_dir = p / "checkpoints"
    ckpt_path = (ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").exists() else (ckpt_dir / "last.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        model.load_state_dict(ckpt["ema_state_dict"])
        src = "EMA"
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        src = "model"
    model.eval()
    print(f"Loaded {src} weights from {ckpt_path.name} (epoch {ckpt.get('epoch', 0) + 1})")

    diffusion = PatchDiffusion(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule_type=config.diffusion.schedule_type,
        device=device,
    )

    return model, diffusion, backbone, config


def load_seed_image(path: str, device: torch.device) -> torch.Tensor:
    """Load seed image as [-1, 1] tensor [3, H, W]."""
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]
    t = t.permute(2, 0, 1) * 2.0 - 1.0                  # [3, H, W] in [-1, 1]
    return t.to(device)


@torch.no_grad()
def generate(
    session_dir: str,
    height: int = 1000,
    width: int = 1000,
    output: str = "generated.png",
    device: str = "cuda",
    fixed_global_context_path: str = None,
    fixed_global_context_fit_path: str = None,
    fixed_full_context_path: str = None,
    text_seed: str = None,
    torch_seed: int = None,
    steps: int = None,
    eta: float = None,
    guidance_scale: float = None,
):
    dev = torch.device(device)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(torch_seed)
        print(f"Random seed: {torch_seed}")
    model, diffusion, backbone, config = load_session(session_dir, dev)
    bs = config.data.trio_block_size

    steps          = steps          or config.diffusion.sampling_steps
    eta            = eta            if eta            is not None else config.diffusion.eta
    guidance_scale = guidance_scale if guidance_scale is not None else config.diffusion.guidance_scale

    # --- Conditioning setup ---
    seed_emb = None
    context_canvas_tensor = None
    orig_h, orig_w = None, None

    if fixed_global_context_fit_path is not None:
        if backbone is None:
            raise ValueError("--fixed-global-context-fit requires a backbone (e.g. CLIP or DINO)")
        seed_img = load_seed_image(fixed_global_context_fit_path, dev)
        _, orig_h, orig_w = seed_img.shape
        height = ((orig_h + bs - 1) // bs) * bs
        width  = ((orig_w + bs - 1) // bs) * bs
        seed_emb = backbone(seed_img.unsqueeze(0))
        print(f"Fixed global context (fit): {fixed_global_context_fit_path} ({orig_h}×{orig_w}) → padded to {height}×{width}")

    elif text_seed is not None:
        if not isinstance(backbone, CLIPBackbone):
            raise ValueError("--text-seed requires a CLIP backbone (config backbone.name='clip')")
        seed_emb = backbone.encode_text(text_seed)
        print(f"Text seed: \"{text_seed}\" → fixed CLIP conditioning throughout")

    elif fixed_global_context_path is not None:
        if backbone is None:
            raise ValueError("--fixed-global-context requires a backbone (e.g. CLIP or DINO)")
        seed_img = load_seed_image(fixed_global_context_path, dev)
        seed_emb = backbone(seed_img.unsqueeze(0))
        print(f"Fixed global context: {fixed_global_context_path} → fixed backbone conditioning throughout")

    elif fixed_full_context_path is not None:
        ctx_img = load_seed_image(fixed_full_context_path, dev)
        _, orig_h, orig_w = ctx_img.shape
        height = ((orig_h + bs - 1) // bs) * bs
        width  = ((orig_w + bs - 1) // bs) * bs
        if backbone is not None:
            seed_emb = backbone(ctx_img.unsqueeze(0))
            print(f"Fixed full context: {fixed_full_context_path} ({orig_h}×{orig_w}) → backbone global + image local patches")
        else:
            print(f"Fixed full context: {fixed_full_context_path} ({orig_h}×{orig_w}) → image local patches only (no backbone)")
        context_canvas_tensor = ctx_img

    elif backbone is not None:
        print("Backbone active: re-encoding canvas progressively")

    # Round down to nearest multiple of patch size
    H = (height // bs) * bs
    W = (width  // bs) * bs

    # Use learned unconditional embedding from model for CFG
    uncond_emb = model.uncond_embedding if hasattr(model, 'uncond_embedding') else None

    gen_rows = H // bs
    gen_cols = W // bs
    print(f"Output: {H}×{W}px | Patches: {gen_rows}×{gen_cols}={gen_rows * gen_cols} @ {bs}×{bs}px")

    use_pos = getattr(config.model, 'use_position_encoding', False)
    use_coord = getattr(config.model, 'use_coordinate_channels', False)
    num_refine = getattr(config.evaluation, 'num_refinement_passes', 0)
    final = autoregressive_generate(
        model, diffusion, backbone, bs, H, W,
        seed_embedding=seed_emb,
        uncond_embedding=uncond_emb,
        steps=steps, eta=eta, guidance_scale=guidance_scale,
        device=dev,
        use_position_encoding=use_pos,
        use_coordinate_channels=use_coord,
        num_refinement_passes=num_refine,
        context_canvas=context_canvas_tensor,
    ).cpu()
    img = (final.clamp(-1, 1) + 1) / 2
    # Crop to original dimensions if a fit/full-context mode was used (remove padding)
    if orig_h is not None and orig_w is not None:
        img = img[:, :orig_h, :orig_w]
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(output)
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image autoregressively from a trained Trio session")
    parser.add_argument("session_dir", help="Path to session directory (contains config.yaml and checkpoints/)")
    parser.add_argument("--size",                     nargs=2, type=int, default=[1000, 1000], metavar=("H", "W"), help="Output size in pixels (default: 1000 1000)")
    parser.add_argument("--output",                   default="generated.png",  help="Output image path")
    parser.add_argument("--device",                   default="cuda",           help="Device (default: cuda)")
    parser.add_argument("--fixed-global-context",     default=None, dest="fixed_global_context",     help="Image encoded by backbone as fixed global conditioning; output at --size (requires backbone)")
    parser.add_argument("--fixed-global-context-fit", default=None, dest="fixed_global_context_fit", help="Image encoded by backbone as fixed global conditioning; output matches image dimensions (requires backbone)")
    parser.add_argument("--fixed-full-context",       default=None, dest="fixed_full_context",       help="Image used for both global backbone conditioning and local patch context; output matches image dimensions")
    parser.add_argument("--text-seed",                default=None, dest="text_seed",                help="Text prompt for CLIP conditioning (requires CLIP backbone)")
    parser.add_argument("--steps",                    type=int,   default=None, help="DDIM steps (default: from config)")
    parser.add_argument("--eta",                      type=float, default=None, help="DDIM eta 0=deterministic (default: from config)")
    parser.add_argument("--guidance-scale",           type=float, default=None, help="CFG guidance scale (default: from config)", dest="guidance_scale")
    parser.add_argument("--torch-seed",               type=int,   default=None, help="Random seed for reproducibility", dest="torch_seed")
    args = parser.parse_args()

    seed_opts = [args.fixed_global_context, args.fixed_global_context_fit, args.fixed_full_context, args.text_seed]
    if sum(x is not None for x in seed_opts) > 1:
        parser.error("--fixed-global-context, --fixed-global-context-fit, --fixed-full-context, and --text-seed are mutually exclusive")

    generate(
        session_dir                    = args.session_dir,
        height                         = args.size[0],
        width                          = args.size[1],
        output                         = args.output,
        device                         = args.device,
        fixed_global_context_path      = args.fixed_global_context,
        fixed_global_context_fit_path  = args.fixed_global_context_fit,
        fixed_full_context_path        = args.fixed_full_context,
        text_seed                      = args.text_seed,
        torch_seed                     = args.torch_seed,
        steps                          = args.steps,
        eta                            = args.eta,
        guidance_scale                 = args.guidance_scale,
    )

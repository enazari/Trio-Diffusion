"""
Trio Training Script

Usage:
    python train.py --config clip              # single GPU
    accelerate launch train.py --config clip   # multi-GPU via DDP
"""

import os
os.environ['OPENBLAS_CORETYPE'] = 'haswell'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import json
import math
import time
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from trio.config import load_config
from trio.data.patch_dataset import PatchDataset
from trio.data.lmdb_cache import ensure_lmdb, NUM_UNSEEN_EVAL
from trio.models.patch_unet import PatchUNet
from trio.diffusion.patch_diffusion import PatchDiffusion, autoregressive_generate, mask_future_patches
from trio.backbones import build_backbone
from trio.backbones.clip import CLIPBackbone
from trio.utils.training_utils import (
    set_seed, save_checkpoint, create_lr_scheduler,
    count_parameters, get_memory_usage, format_time
)
from trio.utils.image_utils import (
    get_context_trio_samples_with_positions,
    apply_flexible_generated_patches, save_comparison_image
)
from trio.utils.ema import EMAModel


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(epoch, model, diffusion, backbone, dataloader, optimizer, scheduler,
                ema, uncond_emb, config, accelerator):
    model.train()
    diffusion.training_mode = True
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.training.epochs}",
                disable=not accelerator.is_main_process)

    for batch in pbar:
        optimizer.zero_grad()
        context_trio = batch['context_trio']
        target_patch = batch['target_patch']

        # Backbone encoding
        global_embedding = None
        if backbone is not None:
            source_image = batch['source_image'].to(accelerator.device, non_blocking=True)
            if config.training.use_teacher_forcing:
                mid = config.training.teacher_forcing_midpoint * config.training.epochs
                temp = config.training.teacher_forcing_temperature * config.training.epochs
                p = 1.0 / (1.0 + math.exp(-(epoch - mid) / max(temp, 1e-6)))
                if torch.rand(1).item() < p:
                    source_image = mask_future_patches(
                        source_image,
                        batch['patch_row'], batch['patch_col'],
                        batch['grid_h'], batch['grid_w'],
                    )
            global_embedding = backbone(source_image)

        # Position encoding
        position = None
        grid_h_t = grid_w_t = None
        use_pos = config.model.use_position_encoding
        use_coord = config.model.use_coordinate_channels
        if use_pos or use_coord:
            grid_h_t = batch['grid_h'].float().to(accelerator.device, non_blocking=True)
            grid_w_t = batch['grid_w'].float().to(accelerator.device, non_blocking=True)
            norm_y = batch['patch_row'].float().to(accelerator.device, non_blocking=True) / grid_h_t.clamp(min=1)
            norm_x = batch['patch_col'].float().to(accelerator.device, non_blocking=True) / grid_w_t.clamp(min=1)
            position = (norm_y, norm_x)

        # Forward
        loss, _, _, _ = diffusion.compute_loss(
            model, target_patch, context_trio,
            global_embedding=global_embedding,
            cfg_dropout_prob=config.diffusion.cfg_dropout_prob,
            uncond_embedding=uncond_emb,
            position=position,
            context_noise_prob=config.training.context_noise_prob,
            context_noise_scale=config.training.context_noise_scale,
            use_coordinate_channels=use_coord,
            grid_h=grid_h_t, grid_w=grid_w_t,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=config.training.gradient_clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update(accelerator.unwrap_model(model))

        val = loss.item()
        if math.isfinite(val):
            total_loss += val
            num_batches += 1

        if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(loss=f"{val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    avg_loss = total_loss / max(num_batches, 1)
    if accelerator.num_processes > 1:
        t = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(t).mean().item()

    return avg_loss


# ---------------------------------------------------------------------------
# Evaluation: sample patches + optional full autoregressive generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(epoch, model, diffusion, backbone, ema, uncond_emb,
             eval_samples, config, device):
    if ema is not None:
        ema.apply(model)
    model.eval()
    diffusion.training_mode = False

    eval_dir = Path(config.evaluation.output_dir) / f"epoch_{epoch + 1:03d}"
    eval_dir.mkdir(exist_ok=True, parents=True)

    bs = config.data.trio_block_size
    use_coord = config.model.use_coordinate_channels

    for img_idx, (original_image, samples, label) in enumerate(eval_samples):
        eval_global_emb = None
        if backbone is not None:
            eval_global_emb = backbone(original_image.unsqueeze(0).to(device))

        _, img_H, img_W = original_image.shape
        grid_h = max(img_H // bs - 1, 1)
        grid_w = max(img_W // bs - 1, 1)

        generated_patches = []
        positions = []
        for context_trio, target_patch, position in samples[:config.evaluation.patches_per_image]:
            ctx_batch = context_trio.unsqueeze(0).to(device)
            tgt_zeros = torch.zeros(1, 3, bs, bs, device=device)
            known_block, mask = diffusion._assemble_block(ctx_batch, tgt_zeros)

            eval_pos = None
            if config.model.use_position_encoding or use_coord:
                row, col = position
                eval_pos = (
                    torch.tensor([row / grid_h], device=device, dtype=torch.float32),
                    torch.tensor([col / grid_w], device=device, dtype=torch.float32),
                )

            coord_maps = None
            if use_coord and eval_pos is not None:
                coord_maps = PatchDiffusion._build_coordinate_maps(
                    eval_pos, bs, 1,
                    grid_h=torch.tensor([grid_h], device=device, dtype=torch.float32),
                    grid_w=torch.tensor([grid_w], device=device, dtype=torch.float32),
                    device=device,
                )

            gen_block, _ = diffusion.ddim_sample(
                model, known_block, mask, (1, 3, 2 * bs, 2 * bs),
                eta=config.diffusion.eta, steps=config.diffusion.sampling_steps,
                global_embedding=eval_global_emb,
                guidance_scale=config.diffusion.guidance_scale,
                uncond_embedding=uncond_emb,
                position=eval_pos, coord_maps=coord_maps,
            )
            generated_patches.append(gen_block[0, :, bs:, bs:].cpu())
            positions.append(position)

        result = apply_flexible_generated_patches(
            original_image, generated_patches, positions, bs)
        if config.evaluation.save_comparisons:
            save_comparison_image(original_image, result,
                                 str(eval_dir / f"comparison_{label}_{img_idx + 1:02d}.png"),
                                 config.data.normalize_range)

    if ema is not None:
        ema.apply(model)
    print(f"  Saved evaluation to {eval_dir}")


@torch.no_grad()
def generate_full_images(epoch, model, diffusion, backbone, ema, uncond_emb,
                         eval_samples, config, device):
    if ema is not None:
        ema.apply(model)
    model.eval()
    diffusion.training_mode = False

    eval_dir = Path(config.evaluation.output_dir) / f"epoch_{epoch + 1:03d}" / "full_generation"
    eval_dir.mkdir(exist_ok=True, parents=True)

    bs = config.data.trio_block_size
    use_pos = config.model.use_position_encoding
    use_coord = config.model.use_coordinate_channels
    num_refine = config.evaluation.num_refinement_passes

    def save_img(tensor, path):
        img = (tensor.clamp(-1, 1) + 1) / 2
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        Image.fromarray(img).save(str(path))

    def gen(seed_emb, name):
        sz = config.evaluation.generation_size
        result = autoregressive_generate(
            model, diffusion, backbone, bs, sz, sz,
            seed_embedding=seed_emb, uncond_embedding=uncond_emb,
            steps=config.diffusion.sampling_steps, eta=config.diffusion.eta,
            guidance_scale=config.diffusion.guidance_scale, device=device,
            use_position_encoding=use_pos, use_coordinate_channels=use_coord,
            num_refinement_passes=num_refine,
        )
        save_img(result, eval_dir / f"{name}.png")

    # Progressive (backbone re-encodes canvas, or no backbone)
    gen(None, "progressive")

    # Fixed seed from eval samples
    seed_train = seed_unseen = None
    for img, _, label in eval_samples:
        if label == "train" and seed_train is None: seed_train = img
        if label == "unseen" and seed_unseen is None: seed_unseen = img

    if backbone is not None and seed_unseen is not None:
        gen(backbone(seed_unseen.unsqueeze(0).to(device)), "fixed_unseen")
    if backbone is not None and seed_train is not None:
        gen(backbone(seed_train.unsqueeze(0).to(device)), "fixed_train")
    if isinstance(backbone, CLIPBackbone):
        gen(backbone.encode_text("cat"), "text_cat")

    if ema is not None:
        ema.apply(model)
    print(f"  Saved full generation to {eval_dir}")


# ---------------------------------------------------------------------------
# Loss CSV
# ---------------------------------------------------------------------------

def save_loss_history(training_log, config):
    if len(training_log) < 2:
        return
    path = Path(config.session_dir) / "loss_history.csv"
    with open(path, 'w') as f:
        f.write("epoch,loss,learning_rate\n")
        for e in training_log:
            f.write(f"{e['epoch']},{e['loss']},{e['learning_rate']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trio training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    mp = "fp16" if config.training.use_mixed_precision else "no"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
    accelerator = Accelerator(mixed_precision=mp, kwargs_handlers=[ddp_kwargs, timeout_kwargs])
    device = accelerator.device

    # --- Data ---
    cache_dir = os.environ.get("TRIO_CACHE_DIR", "datasets")
    lmdb_path = ensure_lmdb(
        cache_dir=cache_dir,
        dataset_path=config.data.dataset_path,
        total_images=config.data.num_images + NUM_UNSEEN_EVAL,
        trio_block_size=config.data.trio_block_size,
        normalize_range=config.data.normalize_range,
        enable_edge_sampling=config.data.enable_edge_sampling,
    )
    dataset = PatchDataset(
        dataset_path=config.data.dataset_path,
        num_images=config.data.num_images,
        trio_block_size=config.data.trio_block_size,
        step_size=config.data.step_size,
        samples_per_epoch=config.data.samples_per_epoch,
        padding_pixels=config.data.padding_pixels,
        eval_samples_per_image=config.data.eval_samples_per_image,
        normalize_range=config.data.normalize_range,
        enable_edge_sampling=config.data.enable_edge_sampling,
        return_source_image=(config.backbone.name != "none"),
        lmdb_path=str(lmdb_path),
    )
    dataloader = DataLoader(
        dataset, batch_size=config.data.batch_size,
        shuffle=True, num_workers=config.data.num_workers, pin_memory=True,
    )

    # Eval samples (main process only)
    eval_samples = []
    if accelerator.is_main_process:
        n = config.evaluation.num_eval_images
        n_train = n // 2
        train_eval = [(img, s, "train") for img, s in dataset.get_fixed_eval_samples(n_train)]
        unseen_eval = [(img, s, "unseen") for img, s in dataset.get_unseen_eval_samples(n - n_train)]
        eval_samples = train_eval + unseen_eval

    # --- Model ---
    backbone = build_backbone(config)
    if backbone is not None:
        backbone = backbone.to(device)

    use_gc = backbone is not None
    model = PatchUNet(
        input_channels=config.model.input_channels,
        output_channels=config.model.output_channels,
        base_channels=config.model.base_channels,
        time_embedding_dim=config.model.time_embedding_dim,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention,
        use_global_conditioning=use_gc,
        global_embedding_dim=config.backbone.embedding_dim if use_gc else 768,
        global_cross_attention_heads=config.backbone.cross_attention_heads if use_gc else 8,
        global_num_tokens=config.backbone.num_tokens if use_gc else 49,
        use_position_encoding=config.model.use_position_encoding,
        use_coordinate_channels=config.model.use_coordinate_channels,
        gate_init=config.training.cross_attention_gate_init,
        backbone_token_pos_enc=config.backbone.use_token_position_encoding,
    ).to(device)
    uncond_emb = model.uncond_embedding if use_gc else None

    # --- Diffusion ---
    diffusion = PatchDiffusion(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule_type=config.diffusion.schedule_type,
        device=device,
    )

    # --- Optimizer + scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                            weight_decay=config.training.weight_decay)
    scheduler = create_lr_scheduler(optimizer, config.training.epochs, config.training.warmup_epochs)

    # --- EMA ---
    ema = EMAModel(model, decay=config.training.ema_decay) if config.training.use_ema else None

    # --- Resume from checkpoint ---
    if config.training.from_checkpoint is not None:
        ckpt = torch.load(config.training.from_checkpoint, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict']
        if any(k.startswith('module.') for k in sd):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        if accelerator.is_main_process:
            print(f"Loaded weights from {config.training.from_checkpoint} (epoch {ckpt.get('epoch', 0) + 1})")

    # --- Accelerate prepare ---
    if config.training.scheduler_type == 'cosine_accelerated':
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if accelerator.is_main_process:
        print(f"Model: {count_parameters(accelerator.unwrap_model(model)):,} parameters")
        if backbone is not None:
            print(f"Backbone: {config.backbone.name} ({config.backbone.embedding_dim}d)")
        print(f"Dataset: {len(dataset)} samples from {len(dataset.image_loader)} images")
        print(f"Session: {config.session_dir}")

    # --- Training loop ---
    training_log = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(config.training.epochs):
        loss = train_epoch(epoch, model, diffusion, backbone, dataloader, optimizer,
                           scheduler, ema, uncond_emb, config, accelerator)
        scheduler.step()

        # Sync all ranks before rank-0 evaluation/checkpointing to prevent
        # NCCL timeout when rank 0 spends a long time on eval/IO.
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            lr = scheduler.get_last_lr()[0]
            training_log.append({'epoch': epoch + 1, 'loss': loss, 'learning_rate': lr})

            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            eta = (elapsed / epochs_done) * (config.training.epochs - epochs_done)
            print(f"Epoch {epoch + 1:3d} | Loss: {loss:.4f} | Time: {format_time(elapsed)} | ETA: {format_time(eta)}")

            is_best = loss < best_loss
            if is_best:
                best_loss = loss

            # Checkpoint
            if (epoch + 1) % config.evaluation.checkpoint_every == 0 or is_best:
                raw = accelerator.unwrap_model(model)
                ema_sd = ema.state_dict() if ema is not None else None
                for name in (["last.pt", "best.pt"] if is_best else ["last.pt"]):
                    save_checkpoint(
                        model=raw, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, loss=loss, config={},
                        checkpoint_path=str(Path(config.checkpoint_dir) / name),
                        ema_state_dict=ema_sd, training_log=training_log,
                    )
                save_loss_history(training_log, config)

            # Evaluate
            if (epoch + 1) % config.evaluation.checkpoint_every == 0:
                evaluate(epoch, accelerator.unwrap_model(model), diffusion, backbone,
                         ema, uncond_emb, eval_samples, config, device)

            # Full AR generation
            gen_every = config.evaluation.generate_full_every
            if gen_every > 0 and (epoch + 1) % gen_every == 0:
                generate_full_images(epoch, accelerator.unwrap_model(model), diffusion,
                                     backbone, ema, uncond_emb, eval_samples, config, device)

        accelerator.wait_for_everyone()

    # Final
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        ema_sd = ema.state_dict() if ema is not None else None
        save_checkpoint(model=raw, optimizer=optimizer, scheduler=scheduler,
                        epoch=config.training.epochs - 1, loss=best_loss, config={},
                        checkpoint_path=str(Path(config.checkpoint_dir) / "last.pt"),
                        ema_state_dict=ema_sd, training_log=training_log)
        evaluate(config.training.epochs - 1, raw, diffusion, backbone,
                 ema, uncond_emb, eval_samples, config, device)
        if config.evaluation.generate_full_every > 0:
            generate_full_images(config.training.epochs - 1, raw, diffusion, backbone,
                                 ema, uncond_emb, eval_samples, config, device)
        print(f"\nDone! Best loss: {best_loss:.4f} | Time: {format_time(time.time() - start_time)}")
        print(f"Session: {config.session_dir}")


if __name__ == "__main__":
    main()

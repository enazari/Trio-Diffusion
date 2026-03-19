"""
Diffusion process for spatial inpainting patch generation.

The model operates on 2×2 blocks: [B, 4, 2*bs, 2*bs] where the 4th channel
is a binary mask (1 = unknown / bottom-right, 0 = known context).
Loss is computed only on the unknown quadrant.
RePaint-style replacement locks known regions at every denoising step.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from tqdm import tqdm


def mask_future_patches(images, patch_rows, patch_cols, grid_hs, grid_ws):
    """Zero out patches at or after (row, col) in raster-scan order.
    Used for teacher forcing: simulates what the backbone sees at inference time."""
    B, C, H, W = images.shape
    masked = images.clone()
    for i in range(B):
        row, col = patch_rows[i].item(), patch_cols[i].item()
        gh, gw = grid_hs[i].item(), grid_ws[i].item()
        ph, pw = H // gh, W // gw
        for r in range(gh):
            for c in range(gw):
                if r > row or (r == row and c >= col):
                    masked[i, :, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = 0
    return masked


class PatchDiffusion:
    """
    Gaussian diffusion with spatial inpainting format and RePaint conditioning.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
        device: str = "cuda"
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.training_mode = True  # Set to False during eval/generation

        if schedule_type == "cosine":
            self.betas, self.alpha_bars = self._cosine_schedule(timesteps)
            self.betas = self.betas.to(self.device)
            self.alpha_bars = self.alpha_bars.to(self.device)
            self.alphas = 1 - self.betas
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(self.device)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        eps = 1e-8

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars + eps)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars + eps)

        self.posterior_variance = self.betas * (1 - self.alpha_bars.roll(1)) / (1 - self.alpha_bars + eps)
        self.posterior_variance[0] = self.betas[0]

        self.sqrt_alpha_bars = torch.clamp(self.sqrt_alpha_bars, min=eps, max=1.0)
        self.sqrt_one_minus_alpha_bars = torch.clamp(self.sqrt_one_minus_alpha_bars, min=eps, max=1.0)

    @staticmethod
    def _cosine_schedule(timesteps: int, s: float = 0.008):
        """Cosine noise schedule (Nichol & Dhariwal 2021)."""
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / timesteps + s) / (1 + s) * (torch.pi / 2)) ** 2
        alpha_bars = f / f[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-6, max=0.999)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas.float(), alpha_bars.float()

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) — forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    @staticmethod
    def _assemble_block(context_trio: torch.Tensor, target_patch: torch.Tensor):
        """Assemble 2×2 spatial block from context trio + target.

        Args:
            context_trio: [B, 9, bs, bs] — TL, TR, BL channel-concatenated
            target_patch: [B, 3, bs, bs]

        Returns:
            block: [B, 3, 2*bs, 2*bs]
            mask:  [B, 1, 2*bs, 2*bs] — 1 for unknown (BR), 0 for known
        """
        B = target_patch.shape[0]
        bs = target_patch.shape[-1]
        dev = target_patch.device

        tl = context_trio[:, 0:3]
        tr = context_trio[:, 3:6]
        bl = context_trio[:, 6:9]

        block = torch.zeros(B, 3, 2 * bs, 2 * bs, device=dev)
        block[:, :, :bs, :bs] = tl
        block[:, :, :bs, bs:] = tr
        block[:, :, bs:, :bs] = bl
        block[:, :, bs:, bs:] = target_patch

        mask = torch.zeros(B, 1, 2 * bs, 2 * bs, device=dev)
        mask[:, :, bs:, bs:] = 1.0

        return block, mask

    @staticmethod
    def _build_coordinate_maps(
        position,
        bs: int,
        batch_size: int,
        grid_h=None,
        grid_w=None,
        device=None,
    ) -> torch.Tensor:
        """Build per-pixel (y, x) coordinate maps for the 2×2 block.

        Args:
            position: (norm_y, norm_x) each [B] in [0, 1)
            bs: patch size (block is 2*bs × 2*bs)
            batch_size: batch size
            grid_h: [B] grid height (used to compute span)
            grid_w: [B] grid width
            device: target device

        Returns:
            [B, 2, 2*bs, 2*bs] coordinate maps
        """
        norm_y, norm_x = position
        if device is None:
            device = norm_y.device

        block_h = 2 * bs
        block_w = 2 * bs

        # Linspace for pixel positions within the block [0, 1]
        y_lin = torch.linspace(0, 1, block_h, device=device)  # [block_h]
        x_lin = torch.linspace(0, 1, block_w, device=device)  # [block_w]

        if grid_h is not None and grid_w is not None:
            # Span of the 2×2 block in normalized image coordinates
            span_y = 2.0 / grid_h.float().clamp(min=1)  # [B]
            span_x = 2.0 / grid_w.float().clamp(min=1)  # [B]
            # Block starts one patch before the target position
            start_y = norm_y - span_y / 2.0  # [B]
            start_x = norm_x - span_x / 2.0  # [B]
            # Map [0,1] linspace to [start, start+span]
            y_maps = start_y.unsqueeze(-1) + y_lin.unsqueeze(0) * span_y.unsqueeze(-1)  # [B, block_h]
            x_maps = start_x.unsqueeze(-1) + x_lin.unsqueeze(0) * span_x.unsqueeze(-1)  # [B, block_w]
        else:
            # Fallback: center on position with unit span
            y_maps = (norm_y.unsqueeze(-1) - 0.5) + y_lin.unsqueeze(0)  # [B, block_h]
            x_maps = (norm_x.unsqueeze(-1) - 0.5) + x_lin.unsqueeze(0)  # [B, block_w]

        # Expand to 2D grids: [B, block_h, block_w]
        y_grid = y_maps.unsqueeze(-1).expand(-1, -1, block_w)
        x_grid = x_maps.unsqueeze(-2).expand(-1, block_h, -1)

        return torch.stack([y_grid, x_grid], dim=1)  # [B, 2, block_h, block_w]

    def compute_loss(
        self,
        model: torch.nn.Module,
        target_patch: torch.Tensor,
        context_trio: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        global_embedding: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.0,
        uncond_embedding: Optional[torch.Tensor] = None,
        position=None,
        context_noise_prob: float = 0.0,
        context_noise_scale: float = 0.2,
        use_coordinate_channels: bool = False,
        grid_h=None,
        grid_w=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss with spatial inpainting format.

        Assembles context_trio + target_patch into a 2×2 block, noises the
        entire block, and computes MSE loss only on the BR quadrant.
        """
        batch_size = target_patch.shape[0]
        bs = target_patch.shape[-1]

        target_patch = torch.clamp(target_patch, -10.0, 10.0)
        context_trio = torch.clamp(context_trio, -10.0, 10.0)

        # Assemble spatial block
        block, mask = self._assemble_block(context_trio, target_patch)

        # Context noise augmentation: add Gaussian noise to known (TL/TR/BL) regions
        if context_noise_prob > 0.0 and self.training_mode:
            noise_mask = torch.rand(batch_size, device=self.device) < context_noise_prob
            if noise_mask.any():
                context_region = 1.0 - mask  # 1 where known, 0 where unknown (BR)
                scale = torch.empty(batch_size, 1, 1, 1, device=self.device).uniform_(
                    0.05, max(context_noise_scale, 0.06)
                )
                ctx_noise = torch.randn_like(block) * scale * context_region
                block = block.clone()
                block[noise_mask] = block[noise_mask] + ctx_noise[noise_mask]

        # CFG dropout: swap global embedding only (keep spatial context intact)
        if cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(batch_size, device=self.device) < cfg_dropout_prob
            if drop_mask.any():
                if global_embedding is not None and uncond_embedding is not None:
                    global_embedding = global_embedding.clone()
                    global_embedding[drop_mask] = uncond_embedding

        # Sample timestep and noise
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        if noise is None:
            noise = torch.randn_like(block)
        noise = torch.clamp(noise, -5.0, 5.0)

        # Noise the entire block
        noisy_block = self.q_sample(block, t, noise)

        # Model input: [noisy_block, mask] → [B, 4, 2*bs, 2*bs]
        model_input = torch.cat([noisy_block, mask], dim=1)

        # Coordinate channels: append (y, x) maps → [B, 6, 2*bs, 2*bs]
        if use_coordinate_channels and position is not None:
            coord_maps = self._build_coordinate_maps(
                position, bs, batch_size, grid_h=grid_h, grid_w=grid_w, device=self.device
            )
            model_input = torch.cat([model_input, coord_maps], dim=1)

        predicted_noise = model(model_input, t, global_embedding=global_embedding, position=position)
        predicted_noise = torch.clamp(predicted_noise, -10.0, 10.0)

        # Loss ONLY on BR quadrant (the unknown region)
        loss = F.mse_loss(
            predicted_noise[:, :, bs:, bs:],
            noise[:, :, bs:, bs:]
        )
        loss = torch.clamp(loss, 0.0, 100.0)

        return loss, predicted_noise, noise, t

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        known_block: torch.Tensor,
        global_embedding: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_embedding: Optional[torch.Tensor] = None,
        position=None,
        coord_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single reverse step p(x_{t-1} | x_t) with RePaint replacement.

        Args:
            x_t: [B, 3, 2*bs, 2*bs] current noisy block
            t: [B] current timestep
            mask: [B, 1, 2*bs, 2*bs] — 1 for unknown, 0 for known
            known_block: [B, 3, 2*bs, 2*bs] clean context block
            coord_maps: optional [B, 2, 2*bs, 2*bs] coordinate channels
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_alphas_t = torch.sqrt(self.alphas[t]).reshape(-1, 1, 1, 1)

        model_input = torch.cat([x_t, mask], dim=1)
        if coord_maps is not None:
            model_input = torch.cat([model_input, coord_maps], dim=1)

        if guidance_scale != 1.0:
            eps_cond = model(model_input, t, global_embedding=global_embedding, position=position)
            eps_uncond = model(model_input, t, global_embedding=uncond_embedding, position=position)
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_theta = model(model_input, t, global_embedding=global_embedding, position=position)

        mean = (1 / sqrt_alphas_t) * (x_t - (betas_t / sqrt_one_minus_alpha_bars_t) * eps_theta)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t]).reshape(-1, 1, 1, 1)
            x_prev = mean + variance * noise

            # RePaint: lock known regions to correctly-noised clean values
            alpha_prev = self.alpha_bars[t[0] - 1]
            repaint_noise = torch.randn_like(x_t)
            known_noised = torch.sqrt(alpha_prev) * known_block + torch.sqrt(1 - alpha_prev) * repaint_noise
            x_prev = mask * x_prev + (1 - mask) * known_noised
        else:
            # Final step: replace known regions with clean values
            x_prev = mask * mean + (1 - mask) * known_block

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        known_block: torch.Tensor,
        mask: torch.Tensor,
        patch_shape: Tuple[int, int, int, int],
        progress: bool = False,
        global_embedding: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_embedding: Optional[torch.Tensor] = None,
        position=None,
        coord_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate via full DDPM sampling with RePaint conditioning."""
        batch_size = known_block.shape[0]

        x_t = torch.randn(patch_shape, device=self.device)
        known_block = known_block.to(self.device)
        mask = mask.to(self.device)

        timesteps = range(self.timesteps - 1, -1, -1)
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(
                model, x_t, t_batch, mask, known_block,
                global_embedding=global_embedding, guidance_scale=guidance_scale,
                uncond_embedding=uncond_embedding, position=position,
                coord_maps=coord_maps,
            )

        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        known_block: torch.Tensor,
        mask: torch.Tensor,
        patch_shape: Tuple[int, int, int, int],
        eta: float = 0.0,
        steps: int = 50,
        progress: bool = False,
        global_embedding: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_embedding: Optional[torch.Tensor] = None,
        position=None,
        coord_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM sampling with RePaint replacement conditioning.

        Args:
            known_block: [B, 3, 2*bs, 2*bs] clean context block (TL, TR, BL filled, BR=0)
            mask: [B, 1, 2*bs, 2*bs] — 1 for unknown (BR), 0 for known
            patch_shape: output shape [B, 3, 2*bs, 2*bs]
            coord_maps: optional [B, 2, 2*bs, 2*bs] coordinate channels
        """
        batch_size = known_block.shape[0]

        timestep_seq = torch.linspace(0, self.timesteps - 1, steps).long().to(self.device)
        timestep_seq = torch.flip(timestep_seq, [0])

        x_t = torch.randn(patch_shape, device=self.device)
        known_block = known_block.to(self.device)
        mask = mask.to(self.device)

        if progress:
            timestep_seq_iter = tqdm(timestep_seq, desc="DDIM Sampling")
        else:
            timestep_seq_iter = timestep_seq

        for i, t in enumerate(timestep_seq_iter):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            model_input = torch.cat([x_t, mask], dim=1)
            if coord_maps is not None:
                model_input = torch.cat([model_input, coord_maps], dim=1)

            # Predict noise (with optional CFG)
            if guidance_scale != 1.0:
                eps_cond = model(model_input, t_batch, global_embedding=global_embedding, position=position)
                eps_uncond = model(model_input, t_batch, global_embedding=uncond_embedding, position=position)
                eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_theta = model(model_input, t_batch, global_embedding=global_embedding, position=position)

            # DDIM update
            alpha_t = self.alpha_bars[t]
            if i < len(timestep_seq) - 1:
                alpha_t_prev = self.alpha_bars[timestep_seq[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)

            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * eps_theta) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)  # x0-clipping

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            noise = torch.randn_like(x_t) if i < len(timestep_seq) - 1 else torch.zeros_like(x_t)

            x_t = (torch.sqrt(alpha_t_prev) * pred_x0
                   + torch.sqrt(1 - alpha_t_prev - sigma_t**2) * eps_theta
                   + sigma_t * noise)

            # RePaint: lock known regions
            if i < len(timestep_seq) - 1:
                alpha_next = self.alpha_bars[timestep_seq[i + 1]]
                repaint_noise = torch.randn_like(x_t)
                known_noised = (torch.sqrt(alpha_next) * known_block
                                + torch.sqrt(1 - alpha_next) * repaint_noise)
                x_t = mask * x_t + (1 - mask) * known_noised
            else:
                # Final step: clean known values
                x_t = mask * x_t + (1 - mask) * known_block

        return x_t, pred_x0

    def get_schedule_info(self) -> dict:
        return {
            'timesteps': self.timesteps,
            'beta_start': float(self.betas[0]),
            'beta_end': float(self.betas[-1]),
            'alpha_bar_min': float(self.alpha_bars.min()),
            'alpha_bar_max': float(self.alpha_bars.max())
        }

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        return pred_x0


@torch.no_grad()
def autoregressive_generate(
    model: torch.nn.Module,
    diffusion: PatchDiffusion,
    backbone,
    block_size: int,
    height: int,
    width: int,
    seed_embedding: Optional[torch.Tensor] = None,
    uncond_embedding: Optional[torch.Tensor] = None,
    steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    device: torch.device = None,
    use_position_encoding: bool = False,
    use_coordinate_channels: bool = False,
    num_refinement_passes: int = 0,
    canvas_init: Optional[torch.Tensor] = None,
    context_canvas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate an image autoregressively in raster-scan order.

    Modes (determined by arguments):
      - seed_embedding provided: fixed conditioning for every patch
      - seed_embedding=None + backbone provided: re-encode canvas each patch
      - backbone=None: no global conditioning
      - context_canvas provided: local TL/TR/BL patches come from this tensor
        instead of the growing generated canvas (used by --fixed-full-context)

    Returns:
        Generated image tensor [3, H, W] in [-1, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    bs = block_size
    H = (height // bs) * bs
    W = (width // bs) * bs

    gen_rows = H // bs
    gen_cols = W // bs

    # Grid dimensions for position encoding — must match training.
    grid_h = max(gen_rows - 1, 1)
    grid_w = max(gen_cols - 1, 1)

    # Canvas with ribbon (one patch wide on top and left)
    if canvas_init is not None:
        import torch.nn.functional as Fgen
        resized = Fgen.interpolate(
            canvas_init.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        )[0].to(device)
        canvas = torch.zeros(3, H + bs, W + bs, device=device)
        canvas[:, bs:, bs:] = resized
        canvas[:, :bs, bs:] = resized[:, :bs, :]
        canvas[:, bs:, :bs] = resized[:, :, :bs]
        canvas[:, :bs, :bs] = resized[:, :bs, :bs]
    else:
        canvas = torch.zeros(3, H + bs, W + bs, device=device)

    # Constant inpainting mask: 1 in BR quadrant
    mask = torch.zeros(1, 1, 2 * bs, 2 * bs, device=device)
    mask[:, :, bs:, bs:] = 1.0

    # Build context_canvas ribbon if a fixed local context image was provided
    ctx_canvas = None
    if context_canvas is not None:
        import torch.nn.functional as Fctx
        resized_ctx = Fctx.interpolate(
            context_canvas.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        )[0].to(device)
        ctx_canvas = torch.zeros(3, H + bs, W + bs, device=device)
        ctx_canvas[:, bs:, bs:] = resized_ctx
        ctx_canvas[:, :bs, bs:] = resized_ctx[:, :bs, :]
        ctx_canvas[:, bs:, :bs] = resized_ctx[:, :, :bs]
        ctx_canvas[:, :bs, :bs] = resized_ctx[:, :bs, :bs]

    total_patches = gen_rows * gen_cols

    def _generate_pass(canvas, seed_emb_override=None, pass_label="Generating"):
        pbar = tqdm(total=total_patches, desc=pass_label)
        for row in range(gen_rows):
            for col in range(gen_cols):
                # Global backbone embedding
                if seed_emb_override is not None:
                    global_emb = seed_emb_override
                elif seed_embedding is not None:
                    global_emb = seed_embedding
                elif backbone is not None:
                    global_emb = backbone(canvas[:, bs:, bs:].clamp(-1, 1).unsqueeze(0))
                else:
                    global_emb = None

                # Extract 2x2 block — local context from ctx_canvas if provided,
                # otherwise from the growing generated canvas
                r, c = row + 1, col + 1
                local_src = ctx_canvas if ctx_canvas is not None else canvas
                block_2x2 = local_src[:, (r - 1) * bs:(r + 1) * bs, (c - 1) * bs:(c + 1) * bs]

                known_block = block_2x2.unsqueeze(0).clone()
                known_block[:, :, bs:, bs:] = 0.0

                # Position encoding
                position = None
                if use_position_encoding or use_coordinate_channels:
                    position = (
                        torch.tensor([row / grid_h], device=device, dtype=torch.float32),
                        torch.tensor([col / grid_w], device=device, dtype=torch.float32),
                    )

                # Coordinate channels
                coord_maps = None
                if use_coordinate_channels and position is not None:
                    coord_maps = PatchDiffusion._build_coordinate_maps(
                        position, bs, 1,
                        grid_h=torch.tensor([grid_h], device=device, dtype=torch.float32),
                        grid_w=torch.tensor([grid_w], device=device, dtype=torch.float32),
                        device=device,
                    )

                output_block, _ = diffusion.ddim_sample(
                    model, known_block, mask, (1, 3, 2 * bs, 2 * bs),
                    eta=eta, steps=steps,
                    global_embedding=global_emb, guidance_scale=guidance_scale,
                    uncond_embedding=uncond_embedding,
                    position=position, coord_maps=coord_maps,
                )

                patch = output_block[0, :, bs:, bs:]
                canvas[:, r * bs:(r + 1) * bs, c * bs:(c + 1) * bs] = patch.clamp(-1, 1)
                pbar.update(1)
        pbar.close()
        return canvas

    canvas = _generate_pass(canvas, pass_label="Generating patches")

    for pass_idx in range(num_refinement_passes):
        refine_emb = None
        if seed_embedding is not None:
            refine_emb = seed_embedding
        elif backbone is not None:
            refine_emb = backbone(canvas[:, bs:, bs:].clamp(-1, 1).unsqueeze(0))
        canvas = _generate_pass(canvas, seed_emb_override=refine_emb,
                                pass_label=f"Refinement {pass_idx + 1}/{num_refinement_passes}")

    return canvas[:, bs:, bs:]

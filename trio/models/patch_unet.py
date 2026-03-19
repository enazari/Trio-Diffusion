"""
PatchUNet model for spatial inpainting diffusion.

Input: [B, 4, 2*bs, 2*bs] — noisy 2×2 block (3 RGB) + binary mask (1 ch)
Output: [B, 3, 2*bs, 2*bs] — predicted noise over entire block
"""

import torch
import torch.nn as nn
from typing import Tuple

from .attention_blocks import SinusoidalPositionEmbeddings
from .residual_blocks import ResidualBlock, DownBlock, UpBlock, MiddleBlock


class PatchUNet(nn.Module):
    """
    UNet for spatial inpainting diffusion.

    Takes a noisy 2×2 block with a binary mask indicating the unknown region,
    and predicts the noise over the entire block. Loss is computed only on the
    unknown (bottom-right) quadrant externally.
    """

    def __init__(
        self,
        input_channels: int = 4,  # 3 RGB + 1 mask
        output_channels: int = 3,
        base_channels: int = 96,
        time_embedding_dim: int = 128,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_global_conditioning: bool = False,
        global_embedding_dim: int = 768,
        global_cross_attention_heads: int = 8,
        global_num_tokens: int = 49,
        num_groups: int = 8,
        use_position_encoding: bool = False,
        use_coordinate_channels: bool = False,
        gate_init: float = 0.0,
        backbone_token_pos_enc: bool = False,
    ):
        super().__init__()

        # If coordinate channels enabled, add 2 input channels for (y, x) maps
        if use_coordinate_channels:
            input_channels = input_channels + 2

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim
        self.use_global_conditioning = use_global_conditioning
        self.use_position_encoding = use_position_encoding
        self.use_coordinate_channels = use_coordinate_channels
        self.backbone_token_pos_enc = backbone_token_pos_enc

        # 2D sinusoidal positional encoding for backbone tokens
        if backbone_token_pos_enc and use_global_conditioning:
            grid_size = int(global_num_tokens ** 0.5)
            pos_enc = self._build_2d_sinusoidal_encoding(
                grid_size, grid_size, global_embedding_dim
            )
            self.register_buffer('backbone_pos_enc', pos_enc)

        # Learned unconditional embedding for CFG (replaces grey-image encoding)
        if use_global_conditioning:
            self.uncond_embedding = nn.Parameter(
                torch.randn(1, global_num_tokens, global_embedding_dim) * 0.02
            )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim)
        )

        # Spatial position encoding: sinusoidal embeddings for (row, col)
        # added to the time embedding so every ResidualBlock is position-aware
        if use_position_encoding:
            self.pos_sin_y = SinusoidalPositionEmbeddings(time_embedding_dim)
            self.pos_sin_x = SinusoidalPositionEmbeddings(time_embedding_dim)
            self.pos_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim * 2, time_embedding_dim * 2),
                nn.SiLU(),
                nn.Linear(time_embedding_dim * 2, time_embedding_dim),
            )

        # Initial convolution: 4 channels (3 RGB + 1 mask) → base_channels
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)

        # Cross-attention kwargs for deeper layers
        ca = dict(
            has_cross_attention=use_global_conditioning,
            cross_attention_dim=global_embedding_dim,
            cross_attention_heads=global_cross_attention_heads,
            gate_init=gate_init,
        )

        # Encoder
        self.down1 = DownBlock(
            base_channels, base_channels * 2, time_embedding_dim,
            has_attention=False, dropout=dropout, num_groups=num_groups
        )
        self.down2 = DownBlock(
            base_channels * 2, base_channels * 4, time_embedding_dim,
            has_attention=False, dropout=dropout, num_groups=num_groups,
            dilated_rf=(4, 8)
        )
        self.down3 = DownBlock(
            base_channels * 4, base_channels * 8, time_embedding_dim,
            has_attention=use_attention, dropout=dropout, num_groups=num_groups, **ca
        )
        self.down4 = DownBlock(
            base_channels * 8, base_channels * 8, time_embedding_dim,
            has_attention=use_attention, dropout=dropout, num_groups=num_groups, **ca
        )

        # Middle
        self.middle = MiddleBlock(
            base_channels * 8, time_embedding_dim, dropout=dropout, num_groups=num_groups, **ca
        )

        # Decoder
        self.up1 = UpBlock(
            base_channels * 8, base_channels * 8, time_embedding_dim,
            has_attention=use_attention, dropout=dropout, num_groups=num_groups, **ca
        )
        self.up2 = UpBlock(
            base_channels * 8, base_channels * 8, time_embedding_dim,
            has_attention=use_attention, dropout=dropout, num_groups=num_groups, **ca
        )
        self.up3 = UpBlock(
            base_channels * 8, base_channels * 4, time_embedding_dim,
            has_attention=False, dropout=dropout, num_groups=num_groups,
            dilated_rf=(4, 8)
        )
        self.up4 = UpBlock(
            base_channels * 4, base_channels * 2, time_embedding_dim,
            has_attention=False, dropout=dropout, num_groups=num_groups
        )

        # Final layers
        self.final_res = ResidualBlock(
            base_channels * 2, base_channels, time_embedding_dim, dropout=dropout, num_groups=num_groups
        )
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(num_groups, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, 3, padding=1)
        )

    @staticmethod
    def _build_2d_sinusoidal_encoding(rows: int, cols: int, dim: int) -> torch.Tensor:
        """Build fixed 2D sinusoidal positional encoding [1, rows*cols, dim]."""
        half = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

        row_pos = torch.arange(rows, dtype=torch.float32)
        col_pos = torch.arange(cols, dtype=torch.float32)

        row_enc = torch.zeros(rows, half)
        row_enc[:, 0::2] = torch.sin(row_pos.unsqueeze(1) * inv_freq.unsqueeze(0))
        row_enc[:, 1::2] = torch.cos(row_pos.unsqueeze(1) * inv_freq.unsqueeze(0))

        col_enc = torch.zeros(cols, half)
        col_enc[:, 0::2] = torch.sin(col_pos.unsqueeze(1) * inv_freq.unsqueeze(0))
        col_enc[:, 1::2] = torch.cos(col_pos.unsqueeze(1) * inv_freq.unsqueeze(0))

        # Combine: for each (row, col) pair, concat row_enc and col_enc
        pe = torch.zeros(rows * cols, dim)
        for r in range(rows):
            for c in range(cols):
                pe[r * cols + c, :half] = row_enc[r]
                pe[r * cols + c, half:] = col_enc[c]

        return pe.unsqueeze(0)  # [1, N, D]

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, global_embedding: torch.Tensor = None, position=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 4, 2*bs, 2*bs] — noisy block (3 ch) + mask (1 ch)
            timestep: [B] diffusion timestep
            global_embedding: optional backbone tokens [B, N, D]
            position: optional (norm_y, norm_x) each [B] in [0, 1)

        Returns:
            Predicted noise [B, 3, 2*bs, 2*bs]
        """
        # Time embeddings
        t_emb = self.time_mlp(timestep)

        # Add spatial position encoding to time embedding
        if self.use_position_encoding and position is not None:
            norm_y, norm_x = position
            y_emb = self.pos_sin_y(norm_y * 1000.0)
            x_emb = self.pos_sin_x(norm_x * 1000.0)
            pos_emb = self.pos_mlp(torch.cat([y_emb, x_emb], dim=-1))
            t_emb = t_emb + pos_emb

        # Initial convolution
        x = self.init_conv(x)

        gc = global_embedding
        if gc is not None and self.backbone_token_pos_enc:
            gc = gc + self.backbone_pos_enc

        # Encoder
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb, gc)
        x, skip4 = self.down4(x, t_emb, gc)

        # Middle
        x = self.middle(x, t_emb, gc)

        # Decoder
        x = self.up1(x, skip4, t_emb, gc)
        x = self.up2(x, skip3, t_emb, gc)
        x = self.up3(x, skip2, t_emb)
        x = self.up4(x, skip1, t_emb)

        # Final layers
        x = self.final_res(x, t_emb)
        x = self.final_conv(x)

        return x

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class LightweightPatchUNet(PatchUNet):
    """Lightweight version of PatchUNet for faster training/inference."""

    def __init__(self, base_channels: int = 64, use_attention: bool = False, **kwargs):
        super().__init__(base_channels=base_channels, use_attention=use_attention, **kwargs)


class HighCapacityPatchUNet(PatchUNet):
    """High-capacity version of PatchUNet for maximum quality."""

    def __init__(self, base_channels: int = 128, time_embedding_dim: int = 256, **kwargs):
        super().__init__(base_channels=base_channels, time_embedding_dim=time_embedding_dim, **kwargs)

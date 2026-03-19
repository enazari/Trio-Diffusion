"""
Residual blocks for the Trio UNet model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_blocks import MultiHeadAttention, GlobalCrossAttentionBlock


class ResidualBlock(nn.Module):
    """Residual block with time embedding integration."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int, 
        dropout: float = 0.1,
        num_groups: int = 8
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_emb_dim: Dimension of time embeddings
            dropout: Dropout probability
            num_groups: Number of groups for group normalization
        """
        super().__init__()
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # First convolution block
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(num_groups, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # Second convolution block
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            time_emb: Time embedding [batch_size, time_emb_dim]
            
        Returns:
            Output tensor [batch_size, out_channels, height, width]
        """
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.block2(h)
        
        # Residual connection
        return h + self.shortcut(x)


class DilatedReceptiveFieldBlock(nn.Module):
    """Depthwise dilated convolutions to expand receptive field across quadrant boundaries."""

    def __init__(self, channels: int, dilations: tuple = (4, 8), num_groups: int = 8):
        super().__init__()
        layers = []
        for d in dilations:
            layers.append(nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels))
            layers.append(nn.GroupNorm(min(num_groups, channels), channels))
            layers.append(nn.SiLU())
        # Pointwise conv to mix channels after depthwise
        layers.append(nn.Conv2d(channels, channels, 1))
        self.net = nn.Sequential(*layers)
        # Zero-init the final pointwise conv so the block starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class DownBlock(nn.Module):
    """Downsampling block with residual connections, optional self-attention and cross-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
        has_cross_attention: bool = False,
        cross_attention_dim: int = 768,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        num_groups: int = 8,
        dilated_rf: tuple = None,
        gate_init: float = 0.0
    ):
        super().__init__()

        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim, dropout, num_groups)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, dropout, num_groups)

        self.dilated_rf_block = DilatedReceptiveFieldBlock(out_channels, dilated_rf, num_groups) if dilated_rf else None

        if has_attention:
            self.attention = MultiHeadAttention(out_channels, num_heads=8, num_groups=num_groups)
        else:
            self.attention = nn.Identity()

        self.cross_attention = (
            GlobalCrossAttentionBlock(out_channels, cross_attention_dim, cross_attention_heads, num_groups, gate_init=gate_init)
            if has_cross_attention else None
        )

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, global_context: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        if self.dilated_rf_block is not None:
            h = self.dilated_rf_block(h)
        h = self.attention(h)
        if self.cross_attention is not None:
            h = self.cross_attention(h, global_context)

        skip = h
        downsampled = self.downsample(h)
        return downsampled, skip


class UpBlock(nn.Module):
    """Upsampling block with residual connections, optional self-attention and cross-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
        has_cross_attention: bool = False,
        cross_attention_dim: int = 768,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        num_groups: int = 8,
        dilated_rf: tuple = None,
        gate_init: float = 0.0
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.res1 = ResidualBlock(out_channels + out_channels, out_channels, time_emb_dim, dropout, num_groups)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, dropout, num_groups)

        self.dilated_rf_block = DilatedReceptiveFieldBlock(out_channels, dilated_rf, num_groups) if dilated_rf else None

        if has_attention:
            self.attention = MultiHeadAttention(out_channels, num_heads=8, num_groups=num_groups)
        else:
            self.attention = nn.Identity()

        self.cross_attention = (
            GlobalCrossAttentionBlock(out_channels, cross_attention_dim, cross_attention_heads, num_groups, gate_init=gate_init)
            if has_cross_attention else None
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, global_context: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)

        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        if self.dilated_rf_block is not None:
            h = self.dilated_rf_block(h)
        h = self.attention(h)
        if self.cross_attention is not None:
            h = self.cross_attention(h, global_context)

        return h


class MiddleBlock(nn.Module):
    """Middle block with self-attention and optional cross-attention for the UNet bottleneck."""

    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        has_cross_attention: bool = False,
        cross_attention_dim: int = 768,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        num_groups: int = 8,
        gate_init: float = 0.0
    ):
        super().__init__()

        self.res1 = ResidualBlock(channels, channels, time_emb_dim, dropout, num_groups)
        self.attention = MultiHeadAttention(channels, num_heads=8, num_groups=num_groups)
        self.cross_attention = (
            GlobalCrossAttentionBlock(channels, cross_attention_dim, cross_attention_heads, num_groups, gate_init=gate_init)
            if has_cross_attention else None
        )
        self.res2 = ResidualBlock(channels, channels, time_emb_dim, dropout, num_groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, global_context: torch.Tensor = None) -> torch.Tensor:
        x = self.res1(x, time_emb)
        x = self.attention(x)
        if self.cross_attention is not None:
            x = self.cross_attention(x, global_context)
        x = self.res2(x, time_emb)
        return x 
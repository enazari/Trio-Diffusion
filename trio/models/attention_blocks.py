"""
Attention mechanisms for the Trio UNet model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time encoding."""
    
    def __init__(self, dim: int):
        """
        Initialize sinusoidal position embeddings.
        
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for time steps.
        
        Args:
            time: Time step tensor [batch_size]
            
        Returns:
            Time embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block for spatial feature refinement."""
    
    def __init__(self, channels: int, num_groups: int = 8):
        """
        Initialize attention block.
        
        Args:
            channels: Number of input channels
            num_groups: Number of groups for group normalization
        """
        super().__init__()
        self.channels = channels
        self.num_groups = min(num_groups, channels)
        
        self.group_norm = nn.GroupNorm(self.num_groups, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to input features.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Attention-refined tensor [batch_size, channels, height, width]
        """
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        # Generate query, key, value
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        
        # Reshape for attention computation
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(B, C, H * W)  # [B, C, HW]
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # Compute attention weights
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Residual connection
        return x + self.proj_out(h)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for enhanced feature processing."""
    
    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 8):
        """
        Initialize multi-head attention.
        
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            num_groups: Number of groups for group normalization
        """
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.group_norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Attention-refined tensor [batch_size, channels, height, width]
        """
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        # Generate query, key, value
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # Each: [B, num_heads, head_dim, HW]
        
        # Compute attention
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, H, W)
        
        # Residual connection
        return x + self.proj_out(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on context patches."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8):
        """
        Initialize cross-attention.

        Args:
            query_dim: Dimension of query features
            context_dim: Dimension of context features
            num_heads: Number of attention heads
        """
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between query and context.

        Args:
            x: Query tensor [batch_size, seq_len, query_dim]
            context: Context tensor [batch_size, context_len, context_dim]

        Returns:
            Attention-refined tensor [batch_size, seq_len, query_dim]
        """
        B, N, D = x.shape

        # Generate query, key, value
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)

        # Residual connection
        return x + self.to_out(out)


class GlobalCrossAttentionBlock(nn.Module):
    """Cross-attention block for global backbone conditioning.

    Takes UNet spatial features [B,C,H,W] and backbone tokens [B,N,D],
    applies cross-attention (Q from UNet, K/V from backbone), returns [B,C,H,W].
    No-op when context is None.
    """

    def __init__(self, channels: int, context_dim: int = 768, num_heads: int = 8, num_groups: int = 8, gate_init: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.to_out = nn.Conv2d(channels, channels, 1)

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            return x

        B, C, H, W = x.shape
        h = self.norm(x)

        # Q from UNet features: [B, C, H, W] -> [B, heads, HW, head_dim]
        q = self.to_q(h).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # K, V from backbone tokens: [B, N, D] -> [B, heads, N, head_dim]
        N = context.shape[1]
        k = self.to_k(context).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: [B, heads, HW, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply: [B, heads, HW, head_dim] -> [B, C, H, W]
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + torch.tanh(self.gate) * self.to_out(out)
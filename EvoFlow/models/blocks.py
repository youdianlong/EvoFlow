import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class AdaGN3D(nn.Module):
    def __init__(self, num_channels, cond_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x, cond):
        x = self.norm(x)
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.ada1 = AdaGN3D(in_ch, cond_dim)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.ada2 = AdaGN3D(out_ch, cond_dim)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.ada1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.ada2(h, cond)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Attention3D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, self.heads, C // self.heads, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, D, H, W)
        return x + self.proj(out)

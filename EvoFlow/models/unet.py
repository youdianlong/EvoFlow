from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.blocks import SinusoidalPosEmb, ResBlock3D, Attention3D


class FlowUNet3D(nn.Module):

    def __init__(
        self,
        num_history: int = 3,
        base_ch: int = 32,
        ch_mults: List[int] = [1, 2, 4, 4],
        cond_dim: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        in_ch = 1 + num_history

        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.cond_fuse = nn.Sequential(
            nn.Linear(time_dim + cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.init_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_ch
        chs = []

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            self.downs.append(nn.ModuleList([
                ResBlock3D(ch, out_ch, cond_dim),
                ResBlock3D(out_ch, out_ch, cond_dim),
                Attention3D(out_ch) if i >= 2 else nn.Identity()
            ]))
            chs.append(out_ch)
            ch = out_ch
            if i < len(ch_mults) - 1:
                self.down_samples.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))

        self.mid1 = ResBlock3D(ch, ch, cond_dim)
        self.mid_attn = Attention3D(ch)
        self.mid2 = ResBlock3D(ch, ch, cond_dim)

        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(ch_mults)):
            out_ch = base_ch * mult
            skip_ch = chs.pop()
            self.ups.append(nn.ModuleList([
                ResBlock3D(ch + skip_ch, out_ch, cond_dim),
                ResBlock3D(out_ch, out_ch, cond_dim),
                Attention3D(out_ch) if (len(ch_mults) - 1 - i) >= 2 else nn.Identity()
            ]))
            ch = out_ch
            if i < len(ch_mults) - 1:
                self.up_samples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    nn.Conv3d(ch, ch, 3, padding=1)
                ))

        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, 1, 3, padding=1)
        )

        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))

    def _forward_block(self, block, x, cond):
        res1, res2, attn = block
        x = res1(x, cond)
        x = res2(x, cond)
        if not isinstance(attn, nn.Identity):
            x = attn(x)
        return x

    def forward(self, z_t, t, cond, history):
        B = z_t.shape[0]

        t_emb = self.time_mlp(t)
        c = self.cond_fuse(torch.cat([t_emb, cond], dim=-1))

        if history.dim() == 6:
            hist = history.squeeze(2)
        else:
            hist = history

        x = torch.cat([z_t, hist], dim=1)
        x = self.init_conv(x)

        skips = []
        for i, block in enumerate(self.downs):
            if self.use_checkpoint and self.training:
                x = checkpoint(self._forward_block, block, x, c, use_reentrant=False)
            else:
                x = self._forward_block(block, x, c)
            skips.append(x)
            if i < len(self.down_samples):
                x = self.down_samples[i](x)

        x = self.mid1(x, c)
        x = self.mid_attn(x)
        x = self.mid2(x, c)

        for i, block in enumerate(self.ups):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            if self.use_checkpoint and self.training:
                x = checkpoint(self._forward_block, block, x, c, use_reentrant=False)
            else:
                x = self._forward_block(block, x, c)
            if i < len(self.up_samples):
                x = self.up_samples[i](x)

        return self.final(x)

    def forward_with_cfg(self, z_t, t, cond, history, cfg_scale=2.0):
        B = z_t.shape[0]
        v_cond = self.forward(z_t, t, cond, history)

        if cfg_scale == 1.0:
            return v_cond

        null_cond = self.null_cond.expand(B, -1)
        v_uncond = self.forward(z_t, t, null_cond, history)

        return v_uncond + cfg_scale * (v_cond - v_uncond)


MODEL_CONFIGS = {
    'S': {'base_ch': 32, 'ch_mults': [1, 2, 4, 4]},
    'M': {'base_ch': 48, 'ch_mults': [1, 2, 4, 4]},
    'L': {'base_ch': 64, 'ch_mults': [1, 2, 4, 8]},
}


def create_model(size='S', num_history=3, cond_dim=256, use_checkpoint=True):
    cfg = MODEL_CONFIGS[size]
    model = FlowUNet3D(
        num_history=num_history,
        base_ch=cfg['base_ch'],
        ch_mults=cfg['ch_mults'],
        cond_dim=cond_dim,
        use_checkpoint=use_checkpoint
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flow UNet ({size}): {n_params/1e6:.2f}M parameters")

    return model

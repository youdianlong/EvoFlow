from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import SinusoidalPosEmb


class DifferenceEncoder3D(nn.Module):

    def __init__(self, num_history: int = 3, feature_dim: int = 128):
        super().__init__()
        self.num_history = num_history
        self.feature_dim = feature_dim
        num_diffs = num_history - 1

        self.diff_encoder = nn.Sequential(
            nn.Conv3d(num_diffs, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim // 4, 1),
            nn.SiLU(),
            nn.Conv3d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, D, H, W = history.shape

        diffs = []
        for t in range(1, T):
            diff = history[:, t] - history[:, t-1]
            diffs.append(diff)
        diff_stack = torch.cat(diffs, dim=1)

        diff_feat = self.diff_encoder(diff_stack)

        attn = self.spatial_attention(diff_feat)
        diff_spatial = diff_feat * attn

        diff_global = self.global_pool(diff_spatial).flatten(1)

        return diff_spatial, diff_global


class VolumeEvolutionEncoder(nn.Module):

    NUM_STRUCTURES = 6

    def __init__(self, num_history: int = 3, evolution_dim: int = 128):
        super().__init__()
        self.num_history = num_history
        self.evolution_dim = evolution_dim

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.NUM_STRUCTURES, 32, kernel_size=2, padding=0),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=2, padding=0) if num_history > 2 else nn.Identity(),
        )

        conv_out_len = num_history - 1
        if num_history > 2:
            conv_out_len = conv_out_len - 1
        conv_out_dim = 64 * max(1, conv_out_len) if num_history > 2 else 32 * conv_out_len

        self.rate_encoder = nn.Sequential(
            nn.Linear(self.NUM_STRUCTURES, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        self.change_encoder = nn.Sequential(
            nn.Linear(self.NUM_STRUCTURES, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        fusion_in = conv_out_dim + 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, evolution_dim),
            nn.LayerNorm(evolution_dim),
            nn.SiLU()
        )

    def forward(self, structure_sequence, evolution_rates, cumulative_change):
        seq_feat = self.temporal_conv(structure_sequence.transpose(1, 2))
        seq_feat = seq_feat.flatten(1)
        rate_feat = self.rate_encoder(evolution_rates)
        change_feat = self.change_encoder(cumulative_change)
        combined = torch.cat([seq_feat, rate_feat, change_feat], dim=-1)
        return self.fusion(combined)


class EvolutionCrossAttention(nn.Module):

    def __init__(self, spatial_dim: int = 128, evolution_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.evolution_to_q = nn.Linear(evolution_dim, spatial_dim)
        self.spatial_to_k = nn.Conv3d(spatial_dim, spatial_dim, 1)
        self.spatial_to_v = nn.Conv3d(spatial_dim, spatial_dim, 1)
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)

        self.norm_evolution = nn.LayerNorm(evolution_dim)
        self.norm_spatial = nn.GroupNorm(8, spatial_dim)

    def forward(self, diff_spatial, evolution_feat):
        B, C, d, h, w = diff_spatial.shape
        N = d * h * w

        evolution_feat = self.norm_evolution(evolution_feat)
        diff_spatial = self.norm_spatial(diff_spatial)

        q = self.evolution_to_q(evolution_feat)
        q = q.view(B, self.num_heads, self.head_dim).unsqueeze(2)

        k = self.spatial_to_k(diff_spatial).flatten(2)
        v = self.spatial_to_v(diff_spatial).flatten(2)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        attn = torch.einsum('bhqd,bhdn->bhqn', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhqn,bhdn->bhqd', attn, v)
        out = out.squeeze(2).reshape(B, C)
        return self.out_proj(out)


class EvolutionGuidedEncoder(nn.Module):

    def __init__(
        self,
        num_history: int = 3,
        out_dim: int = 256,
        diff_feature_dim: int = 128,
        evolution_dim: int = 128,
        frame_feature_dim: int = 64,
        use_difference: bool = True,
        use_evolution: bool = True,
        use_cross_attention: bool = True,
        use_clinical: bool = True,
        use_time_delta: bool = True,
    ):
        super().__init__()

        self.num_history = num_history
        self.out_dim = out_dim
        self.use_difference = use_difference
        self.use_evolution = use_evolution
        self.use_cross_attention = use_cross_attention and use_difference and use_evolution
        self.use_clinical = use_clinical
        self.use_time_delta = use_time_delta

        self.frame_encoder = nn.Sequential(
            nn.Conv3d(1, 16, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(16, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, frame_feature_dim),
            nn.LayerNorm(frame_feature_dim),
            nn.SiLU(),
        )
        frame_feat_dim = frame_feature_dim * num_history

        diff_global_dim = 0
        cross_attn_dim = 0
        if use_difference:
            self.diff_encoder = DifferenceEncoder3D(num_history, diff_feature_dim)
            diff_global_dim = diff_feature_dim
            if self.use_cross_attention:
                self.cross_attention = EvolutionCrossAttention(
                    spatial_dim=diff_feature_dim,
                    evolution_dim=evolution_dim,
                    num_heads=4
                )
                cross_attn_dim = diff_feature_dim

        evolution_out_dim = 0
        if use_evolution:
            self.evolution_encoder = VolumeEvolutionEncoder(num_history, evolution_dim)
            evolution_out_dim = evolution_dim

        time_dim = 0
        if use_time_delta:
            self.time_encoder = nn.Sequential(
                SinusoidalPosEmb(64),
                nn.Linear(64, 128),
                nn.SiLU(),
                nn.Linear(128, 64)
            )
            time_dim = 64

        clinical_dim = 0
        if use_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(8, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 64)
            )
            clinical_dim = 64

        fusion_in_dim = (frame_feat_dim + diff_global_dim + cross_attn_dim +
                        evolution_out_dim + time_dim + clinical_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

        self._init_weights()
        self._print_config(fusion_in_dim, frame_feat_dim, diff_global_dim,
                          cross_attn_dim, evolution_out_dim, time_dim, clinical_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _print_config(self, fusion_in_dim, frame_dim, diff_dim, cross_dim, evo_dim, time_dim, clinical_dim):
        print(f"\n{'='*60}")
        print("Evolution-Guided Encoder Configuration (V2)")
        print(f"{'='*60}")
        print(f"  Difference encoding:    {'ON' if self.use_difference else 'OFF'}")
        print(f"  Evolution encoding:     {'ON' if self.use_evolution else 'OFF'}")
        print(f"  Cross-attention:        {'ON' if self.use_cross_attention else 'OFF'}")
        print(f"  Clinical features:      {'ON' if self.use_clinical else 'OFF'}")
        print(f"  Time delta:             {'ON' if self.use_time_delta else 'OFF'}")
        print(f"{'='*60}")
        print(f"  Feature dimensions:")
        print(f"    Frame features:       {frame_dim} ({frame_dim/fusion_in_dim*100:.1f}%)")
        print(f"    Diff global:          {diff_dim} ({diff_dim/fusion_in_dim*100:.1f}%)")
        print(f"    Cross-attention:      {cross_dim} ({cross_dim/fusion_in_dim*100:.1f}%)")
        print(f"    Evolution:            {evo_dim} ({evo_dim/fusion_in_dim*100:.1f}%)")
        print(f"    Time:                 {time_dim} ({time_dim/fusion_in_dim*100:.1f}%)")
        print(f"    Clinical:             {clinical_dim} ({clinical_dim/fusion_in_dim*100:.1f}%)")
        print(f"{'='*60}")
        print(f"  Fusion input dim:       {fusion_in_dim}")
        print(f"  Output dim:             {self.out_dim}")
        evo_related = diff_dim + cross_dim + evo_dim
        print(f"  Evolution-related:      {evo_related} ({evo_related/fusion_in_dim*100:.1f}%)")
        print(f"{'='*60}\n")

    def forward(
        self,
        history: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
        structure_sequence: Optional[torch.Tensor] = None,
        evolution_rates: Optional[torch.Tensor] = None,
        cumulative_change: Optional[torch.Tensor] = None,
        continuous_features: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history.dim() == 5:
            history = history.unsqueeze(2)

        B, T = history.shape[:2]
        device = history.device
        all_features = []

        frames = history.view(B * T, 1, *history.shape[3:])
        frame_feats = self.frame_encoder(frames).view(B, -1)
        all_features.append(frame_feats)

        diff_spatial = None
        if self.use_difference:
            diff_spatial, diff_global = self.diff_encoder(history)
            all_features.append(diff_global)

        evolution_feat = None
        if self.use_evolution:
            if structure_sequence is not None and evolution_rates is not None and cumulative_change is not None:
                evolution_feat = self.evolution_encoder(
                    structure_sequence, evolution_rates, cumulative_change
                )
            else:
                evolution_feat = torch.zeros(B, 128, device=device)
            all_features.append(evolution_feat)

        if self.use_cross_attention and diff_spatial is not None and evolution_feat is not None:
            cross_attn_feat = self.cross_attention(diff_spatial, evolution_feat)
            all_features.append(cross_attn_feat)

        if self.use_time_delta:
            if time_delta is not None:
                time_feat = self.time_encoder(time_delta.squeeze(-1))
            else:
                time_feat = torch.zeros(B, 64, device=device)
            all_features.append(time_feat)

        if self.use_clinical:
            if continuous_features is not None and categorical_features is not None:
                clinical = torch.cat([continuous_features, categorical_features.float()], dim=-1)
                clinical_feat = self.clinical_encoder(clinical)
            else:
                clinical_feat = torch.zeros(B, 64, device=device)
            all_features.append(clinical_feat)

        combined = torch.cat(all_features, dim=-1)
        return self.fusion(combined)

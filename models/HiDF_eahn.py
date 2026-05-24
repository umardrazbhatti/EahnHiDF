"""
models/HiDF_eahn.py — Explanation-Aware Hybrid Network (EAHN).

v3 patch — mt_std fix:
  EarlyAttnHead tau floor lowered 0.3 → 0.1 so the conv network can produce
  sharper softmax maps earlier in training (uniform map → low mt_std).
  All other architecture unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from HiDF_config import EAHNConfig
from models.HiDF_spatial_stream import SpatialStream
from models.HiDF_temporal_stream import TemporalStream
from models.HiDF_cross_attention import CrossAttentionFusion


@dataclass
class EAHNOutput:
    logit:     torch.Tensor
    prob:      torch.Tensor
    M_t:       torch.Tensor   # (B, T, h, w)
    M_t_up:    torch.Tensor   # (B, T, H, W)
    S:         torch.Tensor   # (B, T, N, d_model)
    low_level: torch.Tensor   # (B, T, C_low, Hl, Wl)
    attn_pool: torch.Tensor   # (B, d_model)


class EarlyAttnHead(nn.Module):
    """Phase 21: produces M_t from CNN feature map BEFORE the transformer.

    v3: tau floor lowered from 0.3 → 0.1.
    Rationale: with tau=0.3 and near-zero conv logits at init, softmax is
    still very uniform (std ≈ 0.02 over 49 cells). Lowering to 0.1 allows
    the gradient to sharpen the map much earlier, directly lifting mt_std.
    The learnable log_tau will grow if the task needs smoother maps.
    """
    def __init__(self, d_model: int = 256, hidden: int = 64,
                 init_temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # Init log_tau to log(0.5) so tau starts at 0.5 (sharper than exp(0)=1)
        self.log_tau = nn.Parameter(torch.tensor(-0.693))  # exp(-0.693) ≈ 0.5

    def forward(self, feats):  # feats: (B, T, C, H, W)
        B, T, C, H, W = feats.shape
        x = feats.reshape(B * T, C, H, W)
        logits = self.proj(x).reshape(B, T, H * W)          # (B, T, H*W)
        # v3: floor lowered 0.3 → 0.1 for sharper maps early in training
        tau = self.log_tau.exp().clamp(min=0.1, max=3.0)
        M = F.softmax(logits / tau, dim=-1)                  # (B, T, H*W), sums to 1
        return M.reshape(B, T, H, W)


class EAHN(nn.Module):
    def __init__(self, config: EAHNConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ── Spatial Stream ────────────────────────────────────────────────────
        self.spatial_stream = SpatialStream(
            backbone_name=config.backbone,
            pretrained=config.backbone_pretrained,
            d_model=d,
            freeze_backbone=False,
        )

        dummy = torch.zeros(1, 3, config.frame_size, config.frame_size)
        with torch.no_grad():
            dummy_tokens = self.spatial_stream(dummy)
        N = dummy_tokens.shape[1]
        self.N      = N
        self.feat_h = self.spatial_stream.feat_h
        self.feat_w = self.spatial_stream.feat_w

        # ── Early Attention Head (Phase 21, v3) ───────────────────────────────
        self.early_attn = EarlyAttnHead(d_model=d, hidden=64)
        self.attn_floor = float(getattr(config, "attn_floor", 0.05))

        # ── Temporal Stream ───────────────────────────────────────────────────
        max_seq = config.num_frames * N + 1
        self.temporal_stream = TemporalStream(
            d_model=d,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout,
            max_seq_len=max_seq,
        )

        # ── Cross-Attention Fusion ────────────────────────────────────────────
        self.cross_attention = CrossAttentionFusion(
            d_model=d,
            num_heads=config.transformer_heads,
            attn_temp_init=getattr(config, "attn_temp_init", 0.0),
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def enable_gradient_checkpointing(self):
        if hasattr(self.temporal_stream, "enable_gradient_checkpointing"):
            self.temporal_stream.enable_gradient_checkpointing()
        if hasattr(self.spatial_stream, "set_grad_checkpointing"):
            self.spatial_stream.set_grad_checkpointing(True)

    def forward(self, frames: torch.Tensor) -> EAHNOutput:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        spatial_tokens = self.spatial_stream(frames_flat)    # (B*T, N, d)
        low_feat = self.spatial_stream.low_level_features()  # (B*T, C_low, Hl, Wl)

        N = spatial_tokens.shape[1]
        d = self.config.d_model
        C_low, Hl, Wl = low_feat.shape[1], low_feat.shape[2], low_feat.shape[3]

        feats_5d = (
            spatial_tokens
            .permute(0, 2, 1)
            .reshape(B * T, d, self.feat_h, self.feat_w)
            .reshape(B, T, d, self.feat_h, self.feat_w)
        )
        M_t_early = self.early_attn(feats_5d)                # (B, T, h, w)
        gate = (M_t_early + self.attn_floor) / (1.0 + self.attn_floor)
        spatial_tokens = (
            feats_5d * gate.unsqueeze(2)
        ).reshape(B * T, d, self.feat_h * self.feat_w).permute(0, 2, 1)

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level      = low_feat.view(B, T, C_low, Hl, Wl)

        Q, cls_out = self.temporal_stream(spatial_tokens.reshape(B, T * N, d))
        Q = Q.reshape(B, T, N, d)

        _legacy_M_t, _legacy_attn_pool = self.cross_attention(Q, spatial_tokens)

        M_flat = M_t_early.reshape(B, T, N)
        attn_pool_per_frame = (Q * M_flat.unsqueeze(-1)).sum(dim=2)
        attn_pool = attn_pool_per_frame.mean(dim=1)

        M_t_up_early = F.interpolate(
            M_t_early.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W), mode="bilinear", align_corners=False,
        ).reshape(B, T, H, W)

        logit = self.classifier(attn_pool).squeeze(-1)
        prob  = torch.sigmoid(logit)

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t_early, M_t_up=M_t_up_early,
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool,
        )

"""
models/eahn.py — Explanation-Aware Hybrid Network (EAHN).

Assembles SpatialStream → TemporalStream → CrossAttentionFusion → classifier.
Single forward pass produces:
  - logit / prob  : classification output
  - M_t           : intrinsic explanation maps  (B, T, h, w) at feature resolution
  - M_t_up        : upsampled explanation maps  (B, T, H, W) for visualisation
  - S             : spatial tokens              (B, T, N, d_model)
  - low_level     : low-level features          (B, T, C_low, Hl, Wl)  for gating
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
    logit:     torch.Tensor   # (B,)
    prob:      torch.Tensor   # (B,)
    M_t:       torch.Tensor   # (B, T, h, w)
    M_t_up:    torch.Tensor   # (B, T, H, W)
    S:         torch.Tensor   # (B, T, N, d_model)
    low_level: torch.Tensor   # (B, T, C_low, Hl, Wl)
    attn_pool: torch.Tensor   # (B, d_model) — attention-weighted pooling for grad path


class EarlyAttnHead(nn.Module):
    """Phase 21: produces M_t from the CNN feature map BEFORE the transformer.

    The map gates features so the transformer (and the classifier) cannot route
    information around it.

    Input  : feats  (B, T, C, H, W)   typically C=d_model, H=W=7
    Output : M_t    (B, T, H, W)      softmax over H*W spatial cells per frame;
                                      each (b, t) plane sums to 1
    """
    def __init__(self, d_model: int = 256, hidden: int = 64,
                 init_temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.log_tau = nn.Parameter(torch.zeros(1))   # learnable temperature; exp(0)=1.0

    def forward(self, feats):  # feats: (B, T, C, H, W)
        B, T, C, H, W = feats.shape
        x = feats.reshape(B * T, C, H, W)
        logits = self.proj(x).reshape(B, T, H * W)    # (B, T, H*W)
        tau = self.log_tau.exp().clamp(min=0.3, max=3.0)
        M = F.softmax(logits / tau, dim=-1)            # (B, T, H*W), sums to 1
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

        # Infer N = h*w from a dummy forward pass
        dummy = torch.zeros(1, 3, config.frame_size, config.frame_size)
        with torch.no_grad():
            dummy_tokens = self.spatial_stream(dummy)
        N = dummy_tokens.shape[1]
        self.N      = N
        self.feat_h = self.spatial_stream.feat_h
        self.feat_w = self.spatial_stream.feat_w

        # ── Early Attention Head (Phase 21) ───────────────────────────────────
        self.early_attn = EarlyAttnHead(
            d_model=d,
            hidden=64,
            init_temperature=1.0,
        )
        self.attn_floor = float(getattr(config, "attn_floor", 0.05))

        # ── Temporal Stream ───────────────────────────────────────────────────
        # max_seq_len = T*N + 1 (CLS token)
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
            self.spatial_stream.set_grad_checkpointing(True)   # timm backbone support

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> EAHNOutput:
        """
        Args:
            frames : (B, T, 3, H, W)
        Returns:
            EAHNOutput
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # Spatial stream — processes all B*T frames in parallel
        spatial_tokens = self.spatial_stream(frames_flat)   # (B*T, N, d)
        low_feat = self.spatial_stream.low_level_features() # (B*T, C_low, Hl, Wl)

        N = spatial_tokens.shape[1]
        d = self.config.d_model
        C_low, Hl, Wl = low_feat.shape[1], low_feat.shape[2], low_feat.shape[3]

        # Phase 21: reshape tokens → (B, T, d, h, w) for EarlyAttnHead conv ops
        feats_5d = (
            spatial_tokens                                               # (B*T, N, d)
            .permute(0, 2, 1)                                            # (B*T, d, N)
            .reshape(B * T, d, self.feat_h, self.feat_w)                 # (B*T, d, h, w)
            .reshape(B, T, d, self.feat_h, self.feat_w)                  # (B, T, d, h, w)
        )
        M_t_early = self.early_attn(feats_5d)                           # (B, T, h, w)
        gate = (M_t_early + self.attn_floor) / (1.0 + self.attn_floor)  # (B, T, h, w)
        # Gate features (broadcast over channel dim), reshape back to token form
        spatial_tokens = (
            feats_5d * gate.unsqueeze(2)                                 # (B, T, d, h, w)
        ).reshape(B * T, d, self.feat_h * self.feat_w).permute(0, 2, 1) # (B*T, N, d)

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level      = low_feat.view(B, T, C_low, Hl, Wl)

        # Temporal stream — flatten T*N gated spatial tokens as the sequence
        Q, cls_out = self.temporal_stream(
            spatial_tokens.reshape(B, T * N, d)
        )                                                    # Q: (B, T*N, d)

        Q = Q.reshape(B, T, N, d)                           # post-transformer spatial tokens

        # Legacy cross-attention block retained; outputs discarded in Phase 21
        _legacy_M_t, _legacy_attn_pool = self.cross_attention(Q, spatial_tokens)

        # Phase 21 Amendment 1: attn_pool from early M_t × post-transformer tokens
        M_flat = M_t_early.reshape(B, T, N)                             # (B, T, N)
        attn_pool_per_frame = (Q * M_flat.unsqueeze(-1)).sum(dim=2)     # (B, T, d)
        attn_pool = attn_pool_per_frame.mean(dim=1)                     # (B, d)

        # Phase 21 Amendment 2: upsample early M_t to input resolution
        M_t_up_early = F.interpolate(
            M_t_early.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, H, W)                               # (B, T, H, W)

        # Classifier reads early M_t-derived attn_pool (Amendment 1)
        final_feat = attn_pool                              # (B, d)
        logit = self.classifier(final_feat).squeeze(-1)     # (B,)
        prob  = torch.sigmoid(logit)

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t_early,       # Phase 21: early M_t (not legacy cross-attn)
            M_t_up=M_t_up_early, # Phase 21: upsample of early M_t
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool, # Phase 21: early M_t × post-transformer tokens
        )

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

from config import EAHNConfig
from models.spatial_stream import SpatialStream
from models.temporal_stream import TemporalStream
from models.cross_attention import CrossAttentionFusion


@dataclass
class EAHNOutput:
    logit:     torch.Tensor   # (B,)
    prob:      torch.Tensor   # (B,)
    M_t:       torch.Tensor   # (B, T, h, w)
    M_t_up:    torch.Tensor   # (B, T, H, W)
    S:         torch.Tensor   # (B, T, N, d_model)
    low_level: torch.Tensor   # (B, T, C_low, Hl, Wl)
    attn_pool: torch.Tensor   # (B, d_model) — attention-weighted pooling for grad path


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

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level      = low_feat.view(B, T, C_low, Hl, Wl)

        # Temporal stream — flatten T*N spatial tokens as the sequence
        Q, cls_out = self.temporal_stream(
            spatial_tokens.reshape(B, T * N, d)
        )                                                    # Q: (B, T*N, d)

        Q = Q.reshape(B, T, N, d)

        # Cross-attention fusion → explanation maps + attention-pooled features
        M_t, attn_pool = self.cross_attention(Q, spatial_tokens)  # (B, T, h, w), (B, d)

        # Upsample explanation maps to input resolution for visualisation / loss
        M_t_up = F.interpolate(
            M_t.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, H, W)                               # (B, T, H, W)

        # Stochastic CLS_out dropout: during training, randomly force classification
        # through the attention branch only, ensuring gradient pressure flows to M_t.
        if self.training and torch.rand(1).item() < self.config.cls_dropout_p:
            final_feat = attn_pool
        else:
            final_feat = cls_out + attn_pool                # (B, d)
        logit = self.classifier(final_feat).squeeze(-1)     # (B,)
        prob  = torch.sigmoid(logit)

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t, M_t_up=M_t_up,
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool,
        )

"""
losses/temporal.py — Gated Temporal Consistency loss L_temp.

w_t = exp(-γ · ||φ(f_t) − φ(f_{t+1})||₂)   where φ is L2-normalised
L_temp = Σ_t w_t · ||M_t − M_{t+1}||²_F  (mean over batch and pairs)

Key fix: low_level features are L2-normalised to unit sphere before computing
pairwise distances, bounding dist ∈ [0, 2] and making γ=0.1 meaningful:
  diff=0.5 → w=exp(-0.05)≈0.95  (similar frames weighted high)
  diff=2.0 → w=exp(-0.20)≈0.82  (very different frames still contribute)
At the old γ=10.0 these gates would be exp(-5) and exp(-20) → ≈ 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        M_t:       torch.Tensor,   # (B, T, h, w)
        low_level: torch.Tensor,   # (B, T, C_low, Hl, Wl)
    ) -> torch.Tensor:
        B, T = M_t.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=M_t.device)

        # Detach, flatten, and L2-normalise to unit sphere
        phi = low_level.detach().reshape(B, T, -1)   # (B, T, C·H'·W')
        phi = F.normalize(phi, p=2, dim=-1)           # unit vectors on d-sphere

        total_loss = torch.tensor(0.0, device=M_t.device)
        n_pairs    = 0

        for t in range(T - 1):
            # L2 distance between consecutive normalised feature vectors; in [0, 2]
            diff_norm = (phi[:, t] - phi[:, t + 1]).norm(dim=-1)   # (B,)

            # Gate: upweights pairs where consecutive frames look similar
            w_t = torch.exp(-self.gamma * diff_norm)               # (B,)

            # Penalise explanation map change weighted by frame similarity
            map_diff = (M_t[:, t] - M_t[:, t + 1]).pow(2).mean(dim=(-1, -2))  # (B,)

            total_loss = total_loss + (w_t * map_diff).mean()
            n_pairs   += 1

        return total_loss / n_pairs

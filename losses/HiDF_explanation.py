"""
losses/explanation.py — L_exp (weakly-supervised only):
  α·Entropy(M_t) + β·TV(M_t) + diversity_weight·l_div

The diversity term penalizes pairwise Jensen-Shannon divergence between heatmaps
from different samples in the batch. The supervised MSE branch has been removed;
this loss is purely intrinsic (no GT masks required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torchvision.transforms import functional as TF


@dataclass
class ExplanationLossOutput:
    loss:            torch.Tensor
    l_h:             float   # entropy term
    l_tv:            float   # total-variation term
    l_div:           float   # inter-sample diversity term
    inter_sample_sim: float  # mean pairwise cosine similarity (diagnostic)


class ExplanationLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, beta: float = 0.5,
                 diversity_weight: float = 2.5):
        super().__init__()
        self.alpha            = alpha
        self.beta             = beta
        self.diversity_weight = diversity_weight

    def forward(
        self,
        M_t: torch.Tensor,   # (B, T, h, w)  normalised to [0,1]
    ) -> ExplanationLossOutput:
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()

        l_h_acc  = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w)

            # Sparsity via entropy
            m_flat  = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
            entropy = -(m_flat * m_flat.log()).sum()

            # Smoothness via total variation
            tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
            tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
            tv   = tv_h + tv_w

            loss     = loss + (self.alpha * entropy + self.beta * tv)
            l_h_acc  += entropy.item()
            l_tv_acc += tv.item()

        loss = loss / B

        # Inter-sample diversity — Jensen-Shannon divergence.
        import math as _math
        N   = B * T
        eye = torch.eye(N, dtype=torch.bool, device=M_t.device)
        n_pairs = N * (N - 1)

        eps = 1e-8
        P = M_t.reshape(N, h * w) + eps
        P = P / P.sum(dim=-1, keepdim=True)

        log_P = P.log()
        P_i   = P.unsqueeze(1)
        P_j   = P.unsqueeze(0)
        M_mix = 0.5 * (P_i + P_j)
        log_M = M_mix.log()
        log_P_i = log_P.unsqueeze(1)
        log_P_j = log_P.unsqueeze(0)
        kl_im = (P_i * (log_P_i - log_M)).sum(dim=-1)
        kl_jm = (P_j * (log_P_j - log_M)).sum(dim=-1)
        js_matrix = 0.5 * (kl_im + kl_jm)

        js_off = js_matrix.masked_fill(eye, 0.0)
        mean_js_tensor = js_off.sum() / max(n_pairs, 1)
        log2 = _math.log(2.0)
        l_div_tensor = (log2 - mean_js_tensor).clamp_min(0.0)
        loss = loss + self.diversity_weight * l_div_tensor

        # Cosine similarity kept as diagnostic.
        flat = M_t.reshape(N, h * w)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_matrix = flat @ flat.T
        inter_sample_sim = float(
            cos_matrix.masked_fill(eye, 0.0).sum().item() / (n_pairs + 1e-8)
        )

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            inter_sample_sim=inter_sample_sim,
        )


# ── Phase 21 utilities ────────────────────────────────────────────────────────

def _gaussian_blur_5d(x: torch.Tensor,
                      kernel_size: int = 21,
                      sigma: float = 10.0) -> torch.Tensor:
    """Gaussian blur over a 5D video tensor.
    x: (B, T, C, H, W) → blurred same shape.
    """
    B, T, C, H, W = x.shape
    x_flat = x.reshape(B * T, C, H, W)
    blurred = TF.gaussian_blur(
        x_flat,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )
    return blurred.reshape(B, T, C, H, W)


def build_bottlenecked_input(x: torch.Tensor,
                              M_t: torch.Tensor,
                              blur_kernel: int = 21,
                              blur_sigma: float = 10.0) -> torch.Tensor:
    """Construct an M_t-gated input at image resolution.

    x   : (B, T, 3, H, W)
    M_t : (B, T, h, w)     softmax over h*w cells per frame (grad preserved)

    Returns x_b (same shape as x):
        x_b = M_norm * x + (1 - M_norm) * blur(x)
    where M_norm is M_t upsampled to (H, W) and rescaled per frame so its
    peak cell maps to 1.0 (raw softmax values are ~1/49, so without this
    rescaling x_b would be nearly all blur).
    """
    B, T, C, H, W = x.shape
    M_up = F.interpolate(
        M_t.reshape(B * T, 1, M_t.shape[-2], M_t.shape[-1]),
        size=(H, W), mode="bilinear", align_corners=False,
    ).reshape(B, T, 1, H, W)

    M_peak = M_up.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    M_norm = (M_up / M_peak).clamp(0.0, 1.0)           # (B, T, 1, H, W)

    with torch.no_grad():
        x_blur = _gaussian_blur_5d(x.detach(), blur_kernel, blur_sigma)

    return M_norm * x + (1.0 - M_norm) * x_blur


def faithfulness_loss(logits_A: torch.Tensor,
                       logits_B: torch.Tensor) -> torch.Tensor:
    """One-way KL from sg(logits_A) to logits_B for binary logits.

    sg(logits_A) is treated as the target; gradient flows only through
    logits_B (and, via x_b, back into M_t and EarlyAttnHead).
    """
    pA = torch.sigmoid(logits_A.detach()).clamp(1e-6, 1.0 - 1e-6)
    pB = torch.sigmoid(logits_B).clamp(1e-6, 1.0 - 1e-6)
    kl = (pA * (pA.log() - pB.log())
          + (1.0 - pA) * ((1.0 - pA).log() - (1.0 - pB).log()))
    return kl.mean()


def sparsity_loss(M_t: torch.Tensor) -> torch.Tensor:
    """Negative mean peak-energy. Minimised when each (b, t) frame has a
    high peak cell. Use a small positive weight (e.g. 0.05) in training.
    """
    return -M_t.amax(dim=(-2, -1)).mean()

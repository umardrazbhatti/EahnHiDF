"""
losses/HiDF_explanation.py — L_exp + faithfulness utilities.

Fix log (v2 — all-three-metrics patch):
  [mt_std fix]        DiversityLoss added: JS-divergence between per-sample
                      M_t distributions penalises samples collapsing to the
                      same spatial mode → mt_std rises.
  [peak_mode_share]   JS-divergence weight raised; PeakSpreadLoss added to
                      directly penalise the fraction of the batch sharing the
                      same argmax location.
  [cosine unchanged]  No changes to ExplanationLoss entropy/TV/JS terms that
                      affect inter_sample_cos (those are already working via
                      the grouped HiDF split in datasets.py).
  [faithfulness fix]  build_bottlenecked_input grad path preserved; B-pass
                      no_grad removed in train_real so loss_faith gradient
                      reaches EarlyAttnHead via x_b → M_norm → M_t_early.
"""

import math as _math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.transforms import functional as TF


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExplanationLossOutput:
    loss:             torch.Tensor
    l_h:              float   # entropy term
    l_tv:             float   # total-variation term
    l_div:            float   # inter-sample JS-divergence term
    inter_sample_sim: float   # mean pairwise cosine similarity (diagnostic)


# ── Main explanation loss (entropy + TV + JS-div) ─────────────────────────────

class ExplanationLoss(nn.Module):
    """Weakly-supervised explanation loss — no GT masks required.

    Terms:
      α · Entropy(M_t)          — encourages sparse, peaked maps
      β · TV(M_t)               — encourages spatially smooth maps
      diversity_weight · L_div  — JS-divergence pushes different samples
                                  to attend to different locations
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.5,
                 diversity_weight: float = 4.0):   # raised from 2.5 → 4.0
        super().__init__()
        self.alpha            = alpha
        self.beta             = beta
        self.diversity_weight = diversity_weight

    def forward(self, M_t: torch.Tensor) -> ExplanationLossOutput:
        """
        M_t : (B, T, h, w)  — softmax maps from EarlyAttnHead, sums to 1 per (b,t)
        """
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()

        l_h_acc  = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w) — temporal average for sample i

            # Entropy — push map toward a single peak (sparse)
            m_flat  = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
            entropy = -(m_flat * m_flat.log()).sum()

            # Total variation — smooth the map spatially
            tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
            tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
            tv   = tv_h + tv_w

            loss     = loss + (self.alpha * entropy + self.beta * tv)
            l_h_acc  += entropy.item()
            l_tv_acc += tv.item()

        loss = loss / B

        # ── Inter-sample JS-divergence (pushes samples to attend differently) ──
        N   = B * T
        eye = torch.eye(N, dtype=torch.bool, device=M_t.device)
        n_pairs = N * (N - 1)

        eps = 1e-8
        P = M_t.reshape(N, h * w) + eps
        P = P / P.sum(dim=-1, keepdim=True)

        log_P   = P.log()
        P_i     = P.unsqueeze(1)                        # (N, 1, hw)
        P_j     = P.unsqueeze(0)                        # (1, N, hw)
        M_mix   = 0.5 * (P_i + P_j)
        log_M   = M_mix.log()
        log_P_i = log_P.unsqueeze(1)
        log_P_j = log_P.unsqueeze(0)

        kl_im = (P_i * (log_P_i - log_M)).sum(dim=-1)  # (N, N)
        kl_jm = (P_j * (log_P_j - log_M)).sum(dim=-1)
        js_matrix = 0.5 * (kl_im + kl_jm)              # (N, N), in [0, log2]

        js_off = js_matrix.masked_fill(eye, 0.0)
        mean_js = js_off.sum() / max(n_pairs, 1)

        # Loss = log(2) - mean_JS  →  minimised when samples maximally differ
        log2        = _math.log(2.0)
        l_div_tensor = (log2 - mean_js).clamp_min(0.0)
        loss = loss + self.diversity_weight * l_div_tensor

        # Cosine similarity — diagnostic only, not in loss
        flat = M_t.reshape(N, h * w)
        flat_n = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_matrix = flat_n @ flat_n.T
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


# ── Peak-spread diversity loss (fixes peak_mode_share) ────────────────────────

class PeakSpreadLoss(nn.Module):
    """Penalises the batch when many samples share the same argmax location.

    peak_mode_share = max_count / B  where max_count is how many samples in
    the batch have the same argmax cell in their time-averaged M_t map.

    We approximate the hard argmax with a soft-max concentration measure so
    the loss remains differentiable:
        L_peak = mean_over_cells( softmax(cell_totals / τ_soft)² )
    This is minimised when cell_totals are uniform (all cells receive equal
    total attention across the batch) and maximised when one cell dominates.

    τ_soft : temperature for the soft concentration; lower = closer to hard
             argmax behaviour. Default 1.0 works well.
    """
    def __init__(self, tau_soft: float = 1.0):
        super().__init__()
        self.tau_soft = tau_soft

    def forward(self, M_t: torch.Tensor) -> torch.Tensor:
        """
        M_t : (B, T, h, w)
        Returns scalar loss.
        """
        B, T, h, w = M_t.shape
        # Time-average per sample → (B, hw)
        m_avg = M_t.mean(dim=1).reshape(B, h * w)

        # Sum across batch → how much each cell is attended to in total (hw,)
        cell_totals = m_avg.sum(dim=0)   # (hw,)

        # Soft concentration: if one cell dominates, softmax is peaked → high loss
        soft_conc = F.softmax(cell_totals / self.tau_soft, dim=0)  # (hw,)

        # Herfindahl-style: sum of squares — high when concentrated
        loss = (soft_conc ** 2).sum() * (h * w)   # scale so uniform → 1.0

        return loss


# ── Phase 21 utilities ────────────────────────────────────────────────────────

def _gaussian_blur_5d(x: torch.Tensor,
                      kernel_size: int = 21,
                      sigma: float = 10.0) -> torch.Tensor:
    """Gaussian blur over a 5D video tensor.
    x: (B, T, C, H, W) → blurred same shape.
    Uses detach so blur doesn't create gradient through the kernel.
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
    M_t : (B, T, h, w)     softmax over h*w cells per frame

    GRADIENT PATH (mt_std fix):
      M_norm is computed from M_t WITH grad. x_b = M_norm*x + (1-M_norm)*blur(x).
      When the B-pass runs WITHOUT no_grad (fixed in train_real.py), the path is:
        loss_faith → logits_B → model(x_b) → x_b → M_norm → M_t → EarlyAttnHead
      This gives EarlyAttnHead a genuine gradient from faithfulness, forcing it to
      produce varied M_t maps that actually change logits_B, increasing mt_std.

    Returns x_b (same shape as x).
    """
    B, T, C, H, W = x.shape

    # Upsample M_t to image resolution — grad preserved through interpolate
    M_up = F.interpolate(
        M_t.reshape(B * T, 1, M_t.shape[-2], M_t.shape[-1]),
        size=(H, W), mode="bilinear", align_corners=False,
    ).reshape(B, T, 1, H, W)                               # (B, T, 1, H, W)

    # Per-frame peak normalisation: raw softmax values are ~1/49, so without
    # this rescaling x_b ≈ all-blur. Peak maps to 1.0, attending cells visible.
    M_peak = M_up.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    M_norm = (M_up / M_peak).clamp(0.0, 1.0)              # (B, T, 1, H, W)

    # Blur is computed from detached x so only M_norm carries the gradient
    with torch.no_grad():
        x_blur = _gaussian_blur_5d(x.detach(), blur_kernel, blur_sigma)

    # x_b: regions M_norm≈1 keep original signal; M_norm≈0 → blurred (masked out)
    x_b = M_norm * x + (1.0 - M_norm) * x_blur            # (B, T, C, H, W)
    return x_b


def faithfulness_loss(logits_A: torch.Tensor,
                       logits_B: torch.Tensor) -> torch.Tensor:
    """One-way KL: sg(A) as target, B as prediction.

    FIXED gradient path (v2):
      logits_A is stop-gradiented (target).
      logits_B carries full gradient back through model(x_b) → x_b → M_norm → M_t.
      This means EarlyAttnHead receives a gradient from this loss.

    Interpretation: model should be more confident on the clean input (A) than
    on the bottlenecked input (B). If M_t is wrong, x_b blurs away true signal
    and logits_B degrades → high loss → M_t corrects itself.
    """
    pA = torch.sigmoid(logits_A.detach()).clamp(1e-6, 1.0 - 1e-6)
    pB = torch.sigmoid(logits_B).clamp(1e-6, 1.0 - 1e-6)
    kl = (pA * (pA.log() - pB.log())
          + (1.0 - pA) * ((1.0 - pA).log() - (1.0 - pB).log()))
    return kl.mean()


def sparsity_loss(M_t: torch.Tensor) -> torch.Tensor:
    """Negative mean peak-energy per (b, t) frame.

    Minimised when each frame has a high-confidence peak cell.
    Use a small positive weight (e.g. 0.05) so this doesn't overwhelm JS-div.
    """
    return -M_t.amax(dim=(-2, -1)).mean()

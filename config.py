"""
config.py — single source of truth for all EAHN hyperparameters.
CLI overrides via argparse; no hardcoded paths anywhere else.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda1: float = 0.3   # L_exp weight (reduced 0.5→0.3 phase7: L_exp no longer stuck at ~3.0, leave room for L_cls to dominate early)
    lambda2: float = 0.2   # L_temp weight (raised 0.1→0.2 phase6: loosen temporal grip)
    alpha: float = 0.3     # entropy weight in weak supervision (raised 0.05→0.3 phase8: real sharpening pressure against the uniform fixed point)
    beta: float = 0.5      # TV weight in weak supervision
    gamma: float = 0.1     # gate decay rate in L_temp (was 10.0 — caused exp→0)
    attn_temp_init: float = 0.0    # τ = exp(0) = 1.0 (was log(4.0)=1.386 → τ=4, over-smoothed attention from init; phase8)
    attn_diversity_weight: float = 5.0  # weight for JS diversity penalty in L_exp (raised 3.0→5.0 phase8: JS has smaller scale than cosine)
    cls_dropout_p: float = 0.0    # phase7: disabled — attn_pool now informative; joint gradient on every step
    label_smoothing: float = 0.0   # label smoothing for BCE/focal loss (removed phase8: uncap probability range; was 0.05)
    max_per_class: int = 0         # if > 0, subsample train set to this many samples per class

    # ── Classification loss ───────────────────────────────────────────────────
    cls_loss_type: str = "bce"   # "bce" | "focal"
    focal_alpha: float = 1.0   # raised 0.25→1.0 phase8: remove gradient-magnitude shrinkage; WeightedRandomSampler handles balance
    focal_gamma: float = 1.0  # lowered 1.5→1.0 phase6: further de-saturate logits for real class

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 4        # T4-safe with AMP+grad_ckpt: B*T=4*16=64 frames; grad_accum_steps=4 → effective 16
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True   # kept for backward compat; use_amp is the authoritative flag
    num_workers: int = 0   # 0 = safe for Kaggle CUDA; increase locally if desired
    use_amp: bool = True           # FP16 automatic mixed precision (T4 supports FP16 not BF16)
    amp_dtype: str = "fp16"        # "fp16" | "bf16"
    grad_checkpoint: bool = True   # gradient checkpointing in TemporalStream to cut VRAM
    clip_grad_norm: float = 1.0    # max gradient norm for clipping

    # ── Evaluation / Visualisation ────────────────────────────────────────────
    eval_after_train: bool = True
    skip_eval: bool = False          # if True, suppress post-training evaluation entirely
    active_manipulation: str = ""           # REQUIRED at CLI; specialist-only mode
    celebdf_root: str = ""                  # path to Celeb-DF v2 dataset root
    celebdf_eval: bool = False              # run Celeb-DF test eval after FF++ test eval
    save_last_checkpoint: bool = False      # Phase 16 leftover; OFF by default
    explanation_suite: bool = True          # run new explanation metrics block after eval
    save_heatmaps: bool = True
    heatmap_samples: int = 20

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn("No GPU found. Switching to CPU with reduced settings.")
                self._apply_cpu_safe_overrides()

    def _apply_cpu_safe_overrides(self):
        self.num_frames = 4
        self.transformer_layers = 2
        self.transformer_heads = 2
        self.batch_size = 2
        self.mixed_precision = False
        self.use_amp = False
        self.grad_checkpoint = False
        self.num_workers = 0
        if "efficientnet_b4" in self.backbone:
            self.backbone = "efficientnet_b0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EAHNConfig":
        cfg = cls()
        for key, val in vars(args).items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        if not cfg.active_manipulation:
            raise ValueError(
                "--active_manipulation is required (specialist-only mode). "
                "Pass one of: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures"
            )
        return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAHN Training and Evaluation")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["synthetic", "ff++", "celeb_df", "dfdc"])
    parser.add_argument("--dataset_compression", type=str, default=None,
                        help="FF++ compression level, e.g. c23 (default) or c40")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker processes. Use 0 on Kaggle to avoid fork errors.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--heatmap_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--eval_after_train", action="store_true", default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--attn_temp_init", type=float, default=None)
    parser.add_argument("--attn_diversity_weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--cls_dropout_p", type=float, default=None)
    parser.add_argument("--cls_loss_type", type=str, default=None,
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["fp16", "bf16"])
    parser.add_argument("--grad_checkpoint", dest="grad_checkpoint", action="store_true", default=None)
    parser.add_argument("--no_grad_checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing applied to BCE/focal loss target (0.05 = maps 0->0.05, 1->0.95)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="If > 0, subsample train set to this many samples per class (balanced 1k/1k)")
    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="If set, skip post-training evaluation (useful for mid-run Kaggle sessions)")
    parser.add_argument("--active_manipulation", type=str, default=None,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Required: specialist manipulation type to train on.")
    parser.add_argument("--celebdf_root", type=str, default=None,
                        help="Path to Celeb-DF v2 dataset root.")
    parser.add_argument("--celebdf_eval", action="store_true", default=None,
                        help="Run Celeb-DF v2 test evaluation after FF++ test eval.")
    parser.add_argument("--save_last_checkpoint", action="store_true", default=None,
                        help="Save last_checkpoint.pth after every epoch (for multi-session resume).")
    parser.add_argument("--explanation_suite", dest="explanation_suite",
                        action="store_true", default=None,
                        help="Run explanation metrics suite after evaluation.")
    parser.add_argument("--no_explanation_suite", dest="explanation_suite",
                        action="store_false")
    return parser.parse_args()

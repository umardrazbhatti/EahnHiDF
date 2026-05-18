"""
scripts/train_real.py — Phase 6 GPU training on FF++/Celeb-DF/DFDC.

Phase 6 changes vs phase 5d:
  - --max_per_class flag for balanced 1k/1k subsampling  (CHANGE 1)
  - WeightedRandomSampler safety net rebuild              (CHANGE 2)
  - 100-batch rolling log (not per-step)                 (CHANGE 3)
  - Per-epoch attention-diversity diagnostic              (CHANGE 4)
  - label_smoothing wired through build_classification_loss (CHANGE 6)
  - loss_curves.png + metric_curves.png +
    training_history.csv emitted at end of training       (CHANGE 12)
"""

import os
import csv as _csv
import dataclasses as _dataclasses
import math
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import EAHNConfig, parse_args
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from models.eahn import EAHN
from losses.classification import build_classification_loss
from losses.explanation import ExplanationLoss
from losses.temporal import TemporalConsistencyLoss
from metrics.detection import DetectionMetrics
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging_utils import Logger


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        cap  = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        print(f"[Device] {name} | CUDA capability sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(
                f"[WARNING] sm_{cap[0]}{cap[1]} is below PyTorch minimum "
                f"(sm_70). Switch Kaggle accelerator to T4. "
                f"Falling back to CPU for MTCNN. AMP disabled."
            )
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds   = DeepfakeDataset(config, "val",   config.dataset_name)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── DataLoader — Regime A: plain shuffle, no sampler ─────────────────────
    # max_per_class cap is now applied inside DeepfakeDataset._build_ffpp()
    # before the split, giving a true 1000:1000 balanced pool (200/type on the
    # fake side). With balanced data the sampler is redundant and its asymmetric
    # per-video exposure (real over-drawn, fake under-drawn) hurt convergence.
    print("[Sampler] Mode=shuffled  (WeightedRandomSampler DISABLED for Regime A)")
    _train_generator = torch.Generator()
    _train_generator.manual_seed(42)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
        generator=_train_generator,
    )
    print(
        f"[DataLoader] batch_size={config.batch_size}  shuffle=True  "
        f"num_workers={config.num_workers}  drop_last=True  generator=seed42"
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
    )
    print(f"[DataLoader val] batch_size={config.batch_size}  shuffle=False  size={len(val_ds)}")

    # ── Multi-batch class-balance smoke check (Regime A) ──────────────────────
    # Under balanced data + shuffle, a single-class first batch occurs ~12% of
    # the time and is not a bug. Use a disposable side-loader (num_workers=0 so
    # it doesn't compete with the training loader) and check 3 batches; only
    # fail if ALL 3 are single-class, which is statistically near-impossible
    # (~0.05%) and would always indicate a broken split.
    _smoke_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=deepfake_collate_fn, num_workers=0,
    )
    _saw_real = _saw_fake = False
    for _i, _sb in enumerate(iter(_smoke_loader)):
        _bl = _sb["label"].cpu().numpy().astype(int)
        _r, _f = int((_bl == 0).sum()), int((_bl == 1).sum())
        print(f"[Smoke] Batch {_i}: real={_r} fake={_f}")
        if _r > 0: _saw_real = True
        if _f > 0: _saw_fake = True
        if _i == 2: break
    del _smoke_loader
    assert _saw_real and _saw_fake, (
        "All 3 inspected batches are single-class. Split or DataLoader broken — "
        "check DeepfakeDataset._split()."
    )
    print("[Smoke] Both classes seen across 3 batches — Regime A loader OK.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    if config.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("[GradCkpt] Gradient checkpointing enabled on TemporalStream.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # AMP — FP16 on T4 (sm_75). BF16 only on Ampere+. Disable on CPU.
    _use_amp = (
        config.use_amp
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    )
    _amp_dtype = torch.float16 if config.amp_dtype == "fp16" else torch.bfloat16
    _dev_str   = device.type
    scaler     = GradScaler(_dev_str, enabled=_use_amp)
    print(f"[AMP] use_amp={_use_amp}  dtype={config.amp_dtype}")

    logger = Logger(config.output_dir)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = -1.0
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt        = load_checkpoint(config.resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0)      # already-completed epoch count (1-indexed)
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"[Resume] Loaded {config.resume_checkpoint}, "
              f"resuming from epoch {start_epoch + 1}  (best_metric={best_metric:.4f})")
    elif config.resume_checkpoint:
        print(f"[Resume] Checkpoint not found at {config.resume_checkpoint!r} — starting fresh.")

    # ── Losses ────────────────────────────────────────────────────────────────
    # CHANGE 6: label_smoothing read from config by build_classification_loss
    cls_loss_fn = build_classification_loss(config)
    print(
        f"[ClsLoss] {cls_loss_fn.__class__.__name__}  "
        f"label_smoothing={getattr(config, 'label_smoothing', 0.0)}"
    )
    exp_loss_fn  = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,
    )
    temp_loss_fn = TemporalConsistencyLoss(gamma=config.gamma)

    ckpt_path = os.path.join(config.output_dir, "best_model.pth")

    # ── CHANGE 12a: epoch-level training history ───────────────────────────────
    history = {
        "epoch":               [],
        "train_total":         [], "train_cls":  [],
        "train_exp":           [], "train_temp": [],
        "val_auc_roc":         [], "val_balanced_acc":      [],
        "val_real_acc":        [], "val_fake_acc":          [],
        "val_inter_sample_cos": [], "val_mt_std":           [],
    }

    # CHANGE 6 (phase7): build a parallel "clean" loader using the val
    # transform (no augmentation) but the train sample list. If the model
    # later reports high train accuracy on the augmented loader but low
    # accuracy on this clean loader, the model is learning the augmentation
    # pattern, not face features.
    from copy import deepcopy
    import torch as _torch
    _clean_ds = deepcopy(train_ds)
    _clean_ds.heavy_aug = False
    # Force every getitem to use the VAL transform (deterministic resize+norm):
    from data.transforms import get_transforms
    _clean_ds.transform = get_transforms("val", config.frame_size)
    # Also disable the heavy-aug branch entirely by setting minority_class to
    # a sentinel that never matches any real label:
    _clean_ds.minority_class = -1
    _clean_loader = DataLoader(
        _clean_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
    )
    print(f"[sanity] clean (unaugmented) train loader built: {len(_clean_ds)} samples")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_batches = len(train_loader)
    epoch_w       = len(str(start_epoch + config.epochs))  # width for log padding

    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        # epoch is 1-indexed: 1 = first ever epoch, 2 = second, etc.
        # config.epochs is always the number of epochs in THIS session.
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── CHANGE 12b: per-epoch loss accumulator ────────────────────────────
        epoch_acc = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

        # ── CHANGE 3: rolling log accumulator ────────────────────────────────
        LOG_EVERY = 200
        run = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

        for batch_idx, batch in enumerate(train_loader):
            frames   = batch["frames"].to(device, non_blocking=True)
            labels   = batch["label"].to(device, non_blocking=True)

            with autocast(_dev_str, enabled=_use_amp, dtype=_amp_dtype):
                out      = model(frames)
                l_cls    = cls_loss_fn(out.logit, labels)
                exp_out  = exp_loss_fn(out.M_t)
                l_exp    = exp_out.loss
                l_temp   = temp_loss_fn(out.M_t, out.low_level)
                _global_step = (epoch - 1) * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 200.0)
                l_total  = l_cls + _lambda1_eff * l_exp + config.lambda2 * l_temp
                loss     = l_total / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── First-batch diagnostics (first epoch of this session only) ──────
            if epoch == start_epoch + 1 and batch_idx == 0:
                print(f"[DIAG] M_t mean={out.M_t.mean():.4f} std={out.M_t.std():.4f}")
                print(f"[DIAG] L_cls={l_cls.item():.6f} L_exp={l_exp.item():.6f} "
                      f"L_temp={l_temp.item():.6f}")
                print(f"[DIAG] attn_temp=exp({model.cross_attention.log_temp.item():.3f})"
                      f"={torch.exp(model.cross_attention.log_temp).item():.3f}")

            # ── Batch balance check every LOG_EVERY steps ─────────────────────
            if (batch_idx + 1) % LOG_EVERY == 0:
                bl = batch["label"].detach().cpu().numpy().astype(int)
                n_real, n_fake = int((bl == 0).sum()), int((bl == 1).sum())
                print(f"[BatchBalance] step={batch_idx+1} real={n_real} fake={n_fake}")

            # ── Accumulate losses ─────────────────────────────────────────────
            _lt = l_total.item()
            _lc = l_cls.item()
            _le = l_exp.item()
            _lp = l_temp.item()

            run["total"] += _lt;  run["cls"] += _lc
            run["exp"]   += _le;  run["temp"] += _lp;  run["n"] += 1

            epoch_acc["total"] += _lt;  epoch_acc["cls"] += _lc
            epoch_acc["exp"]   += _le;  epoch_acc["temp"] += _lp;  epoch_acc["n"] += 1

            # ── CHANGE 3: rolling log (every LOG_EVERY batches) ──────────────
            if (batch_idx + 1) % LOG_EVERY == 0 or (batch_idx + 1) == total_batches:
                n = max(run["n"], 1)
                _tau = model.cross_attention.log_temp.exp().item()
                print(
                    f"[E{epoch:>{epoch_w}} {batch_idx+1:4d}/{total_batches}] "
                    f"total={run['total']/n:.4f}  cls={run['cls']/n:.4f}  "
                    f"exp={run['exp']/n:.4f}  temp={run['temp']/n:.4f}  "
                    f"tau={_tau:.2f}  sim={exp_out.inter_sample_sim:.2f}"
                )
                run = {"total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0, "n": 0}

        scheduler.step()

        # ── CHANGE 12b cont.: store epoch-average train losses ─────────────────
        n = max(epoch_acc["n"], 1)
        history["epoch"].append(epoch)   # 1-indexed epoch number
        history["train_total"].append(epoch_acc["total"] / n)
        history["train_cls"].append(epoch_acc["cls"]   / n)
        history["train_exp"].append(epoch_acc["exp"]   / n)
        history["train_temp"].append(epoch_acc["temp"] / n)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                out    = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        logger.log_scalars("val", metrics, epoch)
        print(
            f"Epoch {epoch:>{epoch_w}}/{start_epoch + config.epochs} | "
            f"Val AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1_at_0.5']:.4f}"
        )

        _val_real_acc = float(metrics.get("real_accuracy",    0.0))
        _val_fake_acc = float(metrics.get("fake_accuracy",    0.0))
        _val_bal_acc  = float(metrics.get("balanced_accuracy", 0.0))
        print(
            f"[ValMetrics] epoch={epoch} "
            f"real_acc={_val_real_acc:.3f} "
            f"fake_acc={_val_fake_acc:.3f} "
            f"balanced_acc={_val_bal_acc:.3f}"
        )

        # ── CHANGE 4: attention-diversity diagnostic on first val batch ────────
        with torch.no_grad():
            diag_batch  = next(iter(val_loader))
            diag_frames = diag_batch["frames"].to(device)
            diag_out    = model(diag_frames)
            mt          = diag_out.M_t.mean(dim=1)          # (B, h, w)
            mt_flat     = mt.reshape(mt.size(0), -1)        # (B, hw)
            mt_norm     = torch.nn.functional.normalize(mt_flat, dim=1)
            cos_mat     = mt_norm @ mt_norm.t()
            B_d         = cos_mat.size(0)
            off_mask    = ~torch.eye(B_d, dtype=torch.bool, device=cos_mat.device)
            off         = cos_mat[off_mask]
            diag_cosine = float(off.mean()) if off.numel() > 0 else 0.0
            diag_std    = float(mt_flat.std(dim=1).mean())
        model.train()
        print(
            f"[Diag] epoch={epoch} "
            f"inter_sample_cos={diag_cosine:.3f}  mt_std={diag_std:.4f}"
        )

        # ── CHANGE 12c: val metrics history ───────────────────────────────────
        history["val_auc_roc"].append(float(metrics.get("auc_roc", float("nan"))))
        history["val_balanced_acc"].append(_val_bal_acc)
        history["val_real_acc"].append(_val_real_acc)
        history["val_fake_acc"].append(_val_fake_acc)
        history["val_inter_sample_cos"].append(diag_cosine)
        history["val_mt_std"].append(diag_std)

        # CHANGE 6 cont.: deterministic-aug train accuracy
        model.eval()
        _clean_probs, _clean_labels = [], []
        with _torch.no_grad():
            for _i, _b in enumerate(_clean_loader):
                if _i * config.batch_size >= 200:   # cap at ~200 samples
                    break
                _f = _b["frames"].to(device)
                _o = model(_f)
                _clean_probs.extend(_o.prob.cpu().tolist())
                _clean_labels.extend(_b["label"].cpu().tolist())
        _clean_probs  = np.array(_clean_probs)
        _clean_labels = np.array(_clean_labels)
        _clean_real_acc = float(((_clean_probs < 0.5) & (_clean_labels == 0)).sum() /
                                max((_clean_labels == 0).sum(), 1))
        _clean_fake_acc = float(((_clean_probs >= 0.5) & (_clean_labels == 1)).sum() /
                                max((_clean_labels == 1).sum(), 1))
        print(f"[sanity] epoch={epoch} train_clean: real_acc={_clean_real_acc:.3f} "
              f"fake_acc={_clean_fake_acc:.3f}  "
              f"(if real_acc is much lower than val real_acc, aug shortcut still live)")
        model.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        val_auc = metrics.get("auc_roc", float("nan"))
        SELECTION_KEY = "balanced_accuracy_at_optimal"
        sel = metrics.get(SELECTION_KEY)
        if sel is None or not np.isfinite(sel):
            print(f"[CheckpointSelect] {SELECTION_KEY} missing/NaN, falling back to auc_roc")
            sel = metrics.get("auc_roc", 0.0)

        if sel > best_metric:
            best_metric = sel
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                            config, ckpt_path)
            print(f"--> Best model saved ({SELECTION_KEY}: {best_metric:.4f}, "
                  f"val_auc_roc={metrics['auc_roc']:.4f})")

        if config.save_last_checkpoint:
            _last_path = os.path.join(config.output_dir, "last_checkpoint.pth")
            _last_tmp  = _last_path + ".tmp"
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric":          best_metric,
                    "config":               _dataclasses.asdict(config),
                },
                _last_tmp,
            )
            os.replace(_last_tmp, _last_path)
            print(f"[Checkpoint] last_checkpoint.pth saved  "
                  f"(epoch={epoch}, best_metric={best_metric:.4f})")

    logger.close()
    print(f"\nTraining complete. Best balanced_accuracy_at_optimal: {best_metric:.4f}")

    # ── CHANGE 12d: end-of-run plots and CSV ──────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save raw history to CSV
        csv_hist = out_path / "training_history.csv"
        with open(csv_hist, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(list(history.keys()))
            w.writerows(zip(*history.values()))
        print(f"[plot] saved {csv_hist}")

        # Plot 1: training loss convergence (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (key, title) in zip(axes.flat, [
            ("train_total", "Total Loss"),
            ("train_cls",   "Classification Loss"),
            ("train_exp",   "Explanation Loss"),
            ("train_temp",  "Temporal Consistency Loss"),
        ]):
            ax.plot(history["epoch"], history[key], marker="o", linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
        fig.suptitle("EAHN — Training Loss Convergence", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "loss_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'loss_curves.png'}")

        # Plot 2: validation metric trajectories (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (keys, title) in zip(axes.flat, [
            (["val_auc_roc"],                         "Val AUC-ROC"),
            (["val_real_acc", "val_fake_acc"],        "Per-class Val Accuracy"),
            (["val_balanced_acc"],                    "Val Balanced Accuracy"),
            (["val_inter_sample_cos", "val_mt_std"],  "Attention Diversity"),
        ]):
            for k in keys:
                ax.plot(history["epoch"], history[k],
                        marker="o", linewidth=2, label=k)
            if "AUC" in title or "Balanced" in title:
                ax.axhline(0.5, color="grey", linestyle="--",
                           alpha=0.5, label="random")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("EAHN — Validation Metric Trajectories", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "metric_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'metric_curves.png'}")

    except Exception as _plot_err:
        print(f"[plot] Warning: could not generate training plots: {_plot_err}")

    if config.eval_after_train:
        from scripts.evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)

"""
scripts/verify_dataset.py — Pre-training dataset sanity check.

Usage:
    python scripts/verify_dataset.py \\
        --data_root /kaggle/input/.../ffpp_data \\
        --active_manipulation Deepfakes

Exits with code 0 if all checks pass, code 1 if any check fails.

Phase 17: specialist-only mode.  --active_manipulation is required because
DeepfakeDataset now loads exactly one manipulation type.
"""

import argparse
import glob
import os
import sys

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def check_directory(label: str, path: str) -> tuple:
    """Return (exists, file_count, sample_filename)."""
    if not os.path.isdir(path):
        return False, 0, ""
    files = glob.glob(os.path.join(path, "*.mp4"))
    sample = os.path.basename(files[0]) if files else ""
    return True, len(files), sample


def main():
    parser = argparse.ArgumentParser(description="Verify FF++ dataset layout")
    parser.add_argument("--data_root", required=True,
                        help="Root directory of the FF++ dataset")
    parser.add_argument("--active_manipulation", required=True,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Specialist manipulation type to verify (Phase 17).")
    args = parser.parse_args()

    data_root   = args.data_root
    compression = "c23"
    MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    failures = []

    # ── 1. Directory table ────────────────────────────────────────────────────
    print("\n=== Directory Check ===")
    header = f"{'Directory':<60} {'Exists':<8} {'Count':<8} {'Sample'}"
    print(header)
    print("-" * len(header))

    real_dir = os.path.join(data_root, "original_sequences", "youtube", compression, "videos")
    exists, count, sample = check_directory("real", real_dir)
    print(f"{real_dir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
    if not exists or count == 0:
        failures.append(f"Real video directory missing or empty: {real_dir}")

    total_fake = 0
    for method in MANIPULATIONS:
        vdir = os.path.join(data_root, "manipulated_sequences", method, compression, "videos")
        exists, count, sample = check_directory(method, vdir)
        print(f"{vdir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
        total_fake += count

    if total_fake == 0:
        failures.append("Zero fake videos found across all manipulation methods.")

    # ── 2. Dataset loading check ──────────────────────────────────────────────
    print("\n=== Dataset Loading Check ===")
    batch = None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import EAHNConfig
        from data.datasets import DeepfakeDataset
        from data.collate import deepfake_collate_fn

        config = EAHNConfig()
        config.data_root            = data_root
        config.active_manipulation  = args.active_manipulation
        config.frame_size           = 224
        config.train_split          = 0.8
        config.val_split            = 0.1
        config.cache_dir            = "/kaggle/working/.face_cache_verify"
        config.device               = "cpu"

        ds = DeepfakeDataset(config, "train", "ff++")

        # Build a balanced subset: 2 real + 2 fake
        real_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 0]
        fake_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 1]

        if len(real_indices) == 0:
            failures.append("Zero real samples found in dataset.")
        if len(fake_indices) == 0:
            failures.append("Zero fake samples found in dataset.")

        balanced_indices = (
            random.sample(real_indices, min(2, len(real_indices))) +
            random.sample(fake_indices, min(2, len(fake_indices)))
        )
        random.shuffle(balanced_indices)

        verify_subset = Subset(ds, balanced_indices)
        loader = DataLoader(
            verify_subset,
            batch_size=len(balanced_indices),
            collate_fn=deepfake_collate_fn,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))

        labels_in_batch = [int(x) for x in batch["label"].tolist()]
        print(f"  Labels in batch : {labels_in_batch}")
        n_real_in_batch = labels_in_batch.count(0)
        n_fake_in_batch = labels_in_batch.count(1)
        print(f"  Real in batch   : {n_real_in_batch}")
        print(f"  Fake in batch   : {n_fake_in_batch}")
        print(f"  Frames shape    : {tuple(batch['frames'].shape)}")
        if n_real_in_batch == 0 or n_fake_in_batch == 0:
            failures.append(
                f"Batch is unbalanced: real={n_real_in_batch}, "
                f"fake={n_fake_in_batch}. Sampler or shuffle may be broken."
            )
    except Exception as exc:
        failures.append(f"Dataset loading raised: {exc}")
        print(f"  ERROR: {exc}")
        import traceback; traceback.print_exc()

    # ── 3. Forward-pass check ─────────────────────────────────────────────────
    print("\n=== Model Forward-Pass Check ===")
    if batch is not None:
        try:
            from models.eahn import EAHN

            model = EAHN(config).to("cpu")
            model.eval()
            with torch.no_grad():
                out = model(batch["frames"])

            print(f"  prob values : {[f'{p:.3f}' for p in out.prob.cpu().tolist()]}")
            mt = out.M_t
            print(f"  M_t shape   : {tuple(mt.shape)}")
            print(f"  M_t min/max : {mt.min():.4f} / {mt.max():.4f}")
            if mt.min() == mt.max():
                failures.append(
                    "M_t is constant (all same value). Model is not "
                    "producing spatial attention. Check cross-attention module."
                )
        except Exception as exc:
            failures.append(f"Forward pass raised: {exc}")
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
    else:
        failures.append(
            "Forward pass skipped — batch was not loaded. "
            "Fix the dataset loading error above first."
        )

    # ── Result ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED — {len(failures)} issue(s) found:")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — dataset is ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
data/collate.py — custom collate function for deepfake batches.
"""

import torch


def deepfake_collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch])               # (B,T,3,H,W)
    labels = torch.tensor([item["label"]    for item in batch],
                           dtype=torch.float32)                            # (B,)
    meta   = [item["meta"] for item in batch]
    return {
        "frames": frames,
        "label":  labels,
        "meta":   meta,
    }

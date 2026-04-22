"""
Training utilities — seeding, checkpointing, evaluation helpers.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """Best-effort reproducibility. Does NOT force cuDNN determinism
    because that costs real training throughput; flip it manually if you
    need bit-identical runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class CheckpointMeta:
    epoch: int
    best_metric: float
    metric_name: str
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_checkpoint(
    path: str | Path,
    *,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    scheduler_state_dict: Optional[Dict[str, Any]] = None,
    meta: Optional[CheckpointMeta] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model_state_dict": model_state_dict}
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if meta is not None:
        payload["meta"] = meta.to_dict()
    if extra:
        payload.update(extra)
    # Save atomically — write to a tmp file, then rename
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)


class EarlyStopper:
    """Stops training when `metric` hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Call once per epoch. Returns True if this epoch's metric is a new best."""
        improved = False
        if self.best is None:
            self.best = metric
            improved = True
        elif self.mode == "min" and metric < self.best - self.min_delta:
            self.best = metric
            improved = True
        elif self.mode == "max" and metric > self.best + self.min_delta:
            self.best = metric
            improved = True

        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


# --- Siamese evaluation ---------------------------------------------------

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity between a (N, D) and b (M, D). Returns (N, M)."""
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = torch.nn.functional.normalize(b, p=2, dim=-1)
    return a @ b.T


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> torch.Tensor:
    """
    Batch-hard triplet loss. For each anchor in the batch:
      - positive   = hardest (farthest) same-label embedding
      - negative   = hardest (closest) different-label embedding

    This mining dominates final accuracy for Siamese networks — which is
    why the original `train_siamese.py`'s "whatever the DataLoader gives
    us" strategy never hit competitive performance.

    Args:
        embeddings: (N, D) — typically unnormalized; we compute
                    euclidean-squared distances in-place.
        labels:     (N,)   — identity ids (integers).
        margin:     triplet margin.
        squared:    if True, use squared euclidean distance (faster,
                    but different numerical regime for the margin).
    """
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings must be (N, D), labels (N,)")
    n = embeddings.size(0)
    if n < 2:
        return embeddings.new_zeros(())

    # Pairwise squared distances
    dot = embeddings @ embeddings.T
    sq = dot.diag().unsqueeze(0)
    dist = sq + sq.T - 2.0 * dot
    dist = dist.clamp(min=0.0)
    if not squared:
        # Avoid sqrt(0) backward instability
        mask_zero = dist.eq(0.0).float()
        dist = (dist + mask_zero * 1e-16).sqrt() * (1.0 - mask_zero)

    # Masks: same-id off-diagonal = positives; different-id = negatives
    labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    pos_mask = labels_eq & ~eye
    neg_mask = ~labels_eq

    # Hardest positive: max distance among positives (or 0 if none)
    masked_pos = dist.masked_fill(~pos_mask, float("-inf"))
    hardest_pos, _ = masked_pos.max(dim=1)
    # For anchors with no positive in batch, hardest_pos is -inf; zero them
    has_pos = pos_mask.any(dim=1)
    hardest_pos = torch.where(has_pos, hardest_pos, torch.zeros_like(hardest_pos))

    # Hardest negative: min distance among negatives (or +inf if none)
    masked_neg = dist.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = masked_neg.min(dim=1)
    has_neg = neg_mask.any(dim=1)

    valid = has_pos & has_neg
    if not valid.any():
        return embeddings.new_zeros(())

    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return loss[valid].mean()


def pair_verification_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate same/different verification on all pairs in a batch.

    Treats pairs with cosine-similarity >= threshold as "same identity"
    and compares against the ground truth label equality.
    Returns {tp, fp, tn, fn, accuracy, precision, recall}.
    """
    emb = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    sims = emb @ emb.T
    n = emb.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)

    pred_same = (sims >= threshold) & ~eye
    true_same = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye

    tp = float((pred_same & true_same).sum().item())
    fp = float((pred_same & ~true_same).sum().item())
    tn = float((~pred_same & ~true_same & ~eye).sum().item())
    fn = float((~pred_same & true_same).sum().item())

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall}

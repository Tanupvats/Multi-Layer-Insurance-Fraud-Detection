"""
Tests for training utilities. These don't need GPU, only pure torch tensors.

Most important: verify batch_hard_triplet_loss picks truly hardest
positive/negative, since this is the single thing we expect to move the
needle on Siamese quality.
"""

from __future__ import annotations

import pytest
import torch

from training.utils import (
    EarlyStopper,
    batch_hard_triplet_loss,
    cosine_similarity_matrix,
    pair_verification_accuracy,
)


# --- EarlyStopper ---------------------------------------------------------

def test_early_stopper_min_mode():
    s = EarlyStopper(patience=2, mode="min")
    assert s.step(1.0) is True   # first is always improvement
    assert s.step(0.9) is True   # better
    assert not s.should_stop
    assert s.step(0.95) is False  # no improvement
    assert not s.should_stop
    assert s.step(0.96) is False  # no improvement — patience exhausted
    assert s.should_stop


def test_early_stopper_max_mode():
    s = EarlyStopper(patience=1, mode="max")
    s.step(0.5)
    s.step(0.7)   # improvement
    s.step(0.6)   # no
    assert s.should_stop


def test_early_stopper_min_delta():
    s = EarlyStopper(patience=2, mode="min", min_delta=0.05)
    s.step(1.0)
    # Improvement of 0.03 is below min_delta — counts as "no improvement"
    assert s.step(0.97) is False
    assert s.step(0.96) is False
    assert s.should_stop


# --- Batch-hard triplet loss ---------------------------------------------

def test_batch_hard_loss_zero_when_perfectly_separated():
    # Two identities, each with 2 embeddings. Same-identity points are
    # co-located; different identities are far apart → loss should be 0.
    emb = torch.tensor([
        [1.0, 0.0], [1.0, 0.0],   # identity 0
        [0.0, 1.0], [0.0, 1.0],   # identity 1
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_batch_hard_loss_positive_when_mixed():
    # Same-identity pair is FARTHER apart than different-identity pair →
    # hardest positive > hardest negative → loss > 0.
    emb = torch.tensor([
        [0.0, 0.0],  # id 0
        [2.0, 0.0],  # id 0 (far from the other id-0)
        [0.5, 0.0],  # id 1 (close to id-0 #0)
        [0.6, 0.0],  # id 1
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    assert float(loss) > 0


def test_batch_hard_loss_handles_single_identity():
    # One identity only — no negatives to mine. Should return 0 without crashing.
    emb = torch.randn(4, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    assert float(loss) == 0.0


def test_batch_hard_loss_picks_hardest_not_random():
    """
    Build a batch where the "easy" negative is very far but a single
    "hard" negative is very close. Confirm the loss reflects the hard one.
    """
    emb = torch.tensor([
        [0.0, 0.0],   # anchor, id=0
        [1.0, 0.0],   # positive, id=0 (dist 1.0)
        [10.0, 0.0],  # easy negative, id=1
        [0.2, 0.0],   # HARD negative, id=1 (closer than positive!)
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    # With the hard negative at 0.2 and hardest positive at 1.0,
    # for the anchor at index 0: hardest_pos = 1.0, hardest_neg = 0.2
    # loss at that anchor = max(1.0 - 0.2 + 0.3, 0) = 1.1
    # Mean over valid anchors (all 4 here) won't exactly equal 1.1 but
    # the loss should be meaningfully > margin. Hard case produces pain.
    assert float(loss) > 0.3


# --- Cosine similarity matrix --------------------------------------------

def test_cosine_similarity_self_is_one():
    a = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    sim = cosine_similarity_matrix(a, a)
    assert sim[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert sim[1, 1].item() == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_is_zero():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    sim = cosine_similarity_matrix(a, b)
    assert abs(sim[0, 0].item()) < 1e-5


# --- Pair verification accuracy ------------------------------------------

def test_pair_verification_all_same_identity():
    # All embeddings are the identical unit vector → all same-id → all pairs
    # predicted same at threshold 0.5. With labels all equal, all true same too.
    emb = torch.ones(4, 8)
    labels = torch.zeros(4, dtype=torch.long)
    m = pair_verification_accuracy(emb, labels, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["fp"] == 0.0


def test_pair_verification_different_ids_far_apart():
    # 4 embeddings pointing in 4 orthogonal directions, 4 distinct ids.
    emb = torch.eye(4)
    labels = torch.tensor([0, 1, 2, 3])
    m = pair_verification_accuracy(emb, labels, threshold=0.5)
    # No pairs predicted same (all cos sim = 0). All true pairs are different.
    # → all predictions correct.
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["tp"] == 0.0
    assert m["fn"] == 0.0

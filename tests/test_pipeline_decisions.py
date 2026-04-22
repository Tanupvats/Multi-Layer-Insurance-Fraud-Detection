"""
Tests for src.pipeline decision logic — the parts that don't need torch.

We deliberately hit `_pose_pair_in_list` and a stand-alone copy of the
verdict ladder so this test suite runs without loading 1 GB of weights
in CI. The end-to-end pipeline is smoke-tested separately with mocked
inferencers in test_pipeline_integration (next file).
"""

from __future__ import annotations

import pytest

from src.pipeline import _pose_pair_in_list
from src.schema import Pose


def test_pose_pair_symmetry():
    pairs = [["LS", "RS"], ["FL", "FR"], ["BL", "BR"]]
    assert _pose_pair_in_list(Pose.LS, Pose.RS, pairs)
    assert _pose_pair_in_list(Pose.RS, Pose.LS, pairs)
    assert _pose_pair_in_list(Pose.FL, Pose.FR, pairs)
    assert _pose_pair_in_list(Pose.BL, Pose.BR, pairs)


def test_pose_pair_rejects_non_mirror():
    pairs = [["LS", "RS"]]
    assert not _pose_pair_in_list(Pose.FS, Pose.BS, pairs)
    assert not _pose_pair_in_list(Pose.LS, Pose.LS, pairs)
    assert not _pose_pair_in_list(Pose.LS, Pose.FL, pairs)


def test_pose_pair_empty_and_malformed():
    assert not _pose_pair_in_list(Pose.LS, Pose.RS, [])
    # Malformed entries are tolerated — just don't match
    assert not _pose_pair_in_list(Pose.LS, Pose.RS, [["LS"]])
    assert not _pose_pair_in_list(Pose.LS, Pose.RS, [["LS", "RS", "FS"]])


# --- Verdict ladder (reimplemented minimally to avoid model imports) -----

class _MockMirror:
    def __init__(self, ran=False, inv=False):
        self.ran = ran
        self.is_likely_inverted = inv


class _MockIdent:
    def __init__(self, ran=False, sim=None, dup=False):
        self.ran = ran
        self.similarity = sim
        self.is_likely_duplicate = dup


def _decide(mirror, ident, review_thr=0.80) -> str:
    verdict = "CLEAN"
    if mirror.is_likely_inverted:
        verdict = "FRAUD"
    if ident.is_likely_duplicate:
        verdict = "FRAUD"
    elif ident.ran and ident.similarity is not None and ident.similarity >= review_thr:
        if verdict == "CLEAN":
            verdict = "SUSPICIOUS"
    if not mirror.ran and not ident.ran and verdict == "CLEAN":
        verdict = "INCONCLUSIVE"
    return verdict


@pytest.mark.parametrize("mirror,ident,expected", [
    (_MockMirror(True, True),  _MockIdent(True, 0.10, False), "FRAUD"),
    (_MockMirror(True, False), _MockIdent(True, 0.99, True),  "FRAUD"),
    (_MockMirror(True, True),  _MockIdent(True, 0.95, True),  "FRAUD"),
    (_MockMirror(True, False), _MockIdent(True, 0.85, False), "SUSPICIOUS"),
    (_MockMirror(True, False), _MockIdent(True, 0.10, False), "CLEAN"),
    (_MockMirror(False),       _MockIdent(True, 0.10, False), "CLEAN"),
    (_MockMirror(False),       _MockIdent(False),             "INCONCLUSIVE"),
    (_MockMirror(False),       _MockIdent(True, 0.99, True),  "FRAUD"),
    (_MockMirror(True, True),  _MockIdent(True, 0.85, False), "FRAUD"),  # FRAUD beats SUSPICIOUS
])
def test_verdict_ladder(mirror, ident, expected):
    assert _decide(mirror, ident) == expected

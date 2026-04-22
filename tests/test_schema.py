"""
Tests for src.schema — typed FraudReport and its parts.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.schema import (
    Flag,
    FraudReport,
    IdentityCheckResult,
    MirrorCheckResult,
    Pose,
    PoseResult,
    Verdict,
)


def test_verdict_values():
    assert {v.value for v in Verdict} == {"CLEAN", "SUSPICIOUS", "FRAUD", "INCONCLUSIVE"}


def test_pose_8_classes():
    assert {p.value for p in Pose} == {"FS", "LS", "RS", "BS", "FL", "FR", "BL", "BR"}


def test_pose_result_confidence_bounds():
    PoseResult(pose=Pose.FS, confidence=0.0)
    PoseResult(pose=Pose.FS, confidence=1.0)
    with pytest.raises(Exception):
        PoseResult(pose=Pose.FS, confidence=1.5)
    with pytest.raises(Exception):
        PoseResult(pose=Pose.FS, confidence=-0.1)


def test_identity_similarity_bounds():
    IdentityCheckResult(ran=True, similarity=-1.0)
    IdentityCheckResult(ran=True, similarity=1.0)
    with pytest.raises(Exception):
        IdentityCheckResult(ran=True, similarity=1.5)


def test_fraud_report_defaults_have_unran_checks():
    r = FraudReport(
        claim_id="x", generated_at=datetime.now(timezone.utc), verdict=Verdict.CLEAN,
    )
    assert r.mirror_check.ran is False
    assert r.identity_check.ran is False
    assert r.flags == []
    assert r.notes == []


def test_fraud_report_roundtrip_json():
    r = FraudReport(
        claim_id="CLM-1",
        generated_at=datetime.now(timezone.utc),
        verdict=Verdict.FRAUD,
        flags=[Flag.INVERTED_IMAGE_DETECTED, Flag.DUPLICATE_VEHICLE_IDENTITY],
        pose_a=PoseResult(pose=Pose.LS, confidence=0.95),
        pose_b=PoseResult(pose=Pose.RS, confidence=0.93),
        mirror_check=MirrorCheckResult(ran=True, match_count=120, threshold=80,
                                       is_likely_inverted=True),
        identity_check=IdentityCheckResult(ran=True, similarity=0.97, threshold=0.92,
                                            review_threshold=0.8, is_likely_duplicate=True),
        input_a_sha256="a" * 64,
        input_b_sha256="b" * 64,
        duration_ms=1234,
    )
    j = r.model_dump_json()
    back = FraudReport.model_validate_json(j)
    assert back.verdict is Verdict.FRAUD
    assert Flag.INVERTED_IMAGE_DETECTED in back.flags
    assert back.mirror_check.match_count == 120
    assert back.identity_check.similarity == pytest.approx(0.97)

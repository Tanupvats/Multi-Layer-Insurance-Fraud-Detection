"""Tests for api.audit — SQLite round-trip, filtering, latest-wins."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.audit import AuditStore
from src.schema import FraudReport, Verdict


def _make_report(claim_id: str, verdict: Verdict = Verdict.CLEAN) -> FraudReport:
    return FraudReport(
        claim_id=claim_id,
        generated_at=datetime.now(timezone.utc),
        verdict=verdict,
        duration_ms=42,
        input_a_sha256="a" * 64,
        input_b_sha256="b" * 64,
    )


def test_empty_store_count_zero(tmp_path):
    s = AuditStore(tmp_path / "a.db")
    assert s.count() == 0


def test_save_and_retrieve_by_claim_id(tmp_path):
    s = AuditStore(tmp_path / "a.db")
    rid = s.save(_make_report("CLM-1", Verdict.FRAUD))
    assert rid > 0
    back = s.get_by_claim_id("CLM-1")
    assert back is not None
    assert back.claim_id == "CLM-1"
    assert back.verdict is Verdict.FRAUD


def test_missing_claim_returns_none(tmp_path):
    s = AuditStore(tmp_path / "a.db")
    assert s.get_by_claim_id("nope") is None


def test_list_recent_ordering_and_filter(tmp_path):
    s = AuditStore(tmp_path / "a.db")
    for i, v in enumerate([Verdict.CLEAN, Verdict.FRAUD, Verdict.SUSPICIOUS, Verdict.FRAUD]):
        s.save(_make_report(f"CLM-{i:03d}", v))

    recent = s.list_recent(limit=10)
    assert len(recent) == 4
    # Most recent saved first (descending id)
    assert recent[0]["claim_id"] == "CLM-003"

    frauds = s.list_recent(limit=10, verdict="FRAUD")
    assert len(frauds) == 2
    assert all(r["verdict"] == "FRAUD" for r in frauds)


def test_latest_write_wins_for_same_claim_id(tmp_path):
    s = AuditStore(tmp_path / "a.db")
    s.save(_make_report("CLM-9", Verdict.FRAUD))
    s.save(_make_report("CLM-9", Verdict.CLEAN))
    back = s.get_by_claim_id("CLM-9")
    assert back is not None and back.verdict is Verdict.CLEAN


def test_persistence_across_reopens(tmp_path):
    path = tmp_path / "persist.db"
    s1 = AuditStore(path)
    s1.save(_make_report("CLM-1"))
    assert s1.count() == 1
    s2 = AuditStore(path)
    assert s2.count() == 1
    assert s2.get_by_claim_id("CLM-1") is not None

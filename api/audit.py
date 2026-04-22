

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from src.schema import FraudReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claim_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        TEXT    NOT NULL,
    request_id      TEXT,
    verdict         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,   -- ISO 8601 UTC
    input_a_sha256  TEXT,
    input_b_sha256  TEXT,
    duration_ms     INTEGER,
    report_json     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_id   ON claim_analyses(claim_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON claim_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_verdict    ON claim_analyses(verdict);
"""


class AuditStore:
    """
    Thread-safe SQLite wrapper.

    FastAPI with uvicorn --workers=1 is effectively single-threaded for
    our purposes (async requests, single event loop), but tests and
    background tasks can touch the DB — the lock keeps us honest.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            con.row_factory = sqlite3.Row
            yield con
        finally:
            con.close()

    def save(self, report: FraudReport) -> int:
        payload = report.model_dump_json()
        with self._lock, self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO claim_analyses
                    (claim_id, request_id, verdict, created_at,
                     input_a_sha256, input_b_sha256, duration_ms, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.claim_id,
                    report.request_id,
                    report.verdict.value,
                    report.generated_at.astimezone(timezone.utc).isoformat(),
                    report.input_a_sha256,
                    report.input_b_sha256,
                    report.duration_ms,
                    payload,
                ),
            )
            return int(cur.lastrowid)

    def get_by_claim_id(self, claim_id: str) -> Optional[FraudReport]:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT report_json FROM claim_analyses WHERE claim_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (claim_id,),
            ).fetchone()
        if row is None:
            return None
        return FraudReport.model_validate_json(row["report_json"])

    def list_recent(self, limit: int = 50, verdict: Optional[str] = None) -> List[dict]:
        """Light index listing (no full JSON) for dashboards."""
        sql = (
            "SELECT claim_id, verdict, created_at, duration_ms, input_a_sha256, input_b_sha256 "
            "FROM claim_analyses"
        )
        params: tuple = ()
        if verdict:
            sql += " WHERE verdict = ?"
            params = (verdict,)
        sql += " ORDER BY id DESC LIMIT ?"
        params = (*params, int(limit))
        with self._lock, self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT COUNT(*) AS n FROM claim_analyses").fetchone()
        return int(row["n"])

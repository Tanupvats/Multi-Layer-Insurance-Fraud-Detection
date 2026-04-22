

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"  # above review threshold, below fraud threshold
    FRAUD = "FRAUD"
    INCONCLUSIVE = "INCONCLUSIVE"  # couldn't run a check (missing crop, model unavailable, etc.)


class Flag(str, Enum):
    INVERTED_IMAGE_DETECTED = "INVERTED_IMAGE_DETECTED"
    DUPLICATE_VEHICLE_IDENTITY = "DUPLICATE_VEHICLE_IDENTITY"
    POSE_MISMATCH_UNSUPPORTED = "POSE_MISMATCH_UNSUPPORTED"
    WINDSHIELD_NOT_DETECTED = "WINDSHIELD_NOT_DETECTED"
    CAR_NOT_DETECTED = "CAR_NOT_DETECTED"


class Pose(str, Enum):
    """8-way vehicle orientation."""
    FS = "FS"   # Front Straight
    LS = "LS"   # Left Side
    RS = "RS"   # Right Side
    BS = "BS"   # Back Straight
    FL = "FL"   # Front-Left 3/4
    FR = "FR"   # Front-Right 3/4
    BL = "BL"   # Back-Left 3/4
    BR = "BR"   # Back-Right 3/4


class PoseResult(BaseModel):
    pose: Pose
    confidence: float = Field(ge=0.0, le=1.0)
    all_probabilities: Dict[str, float] = Field(default_factory=dict)


class MirrorCheckResult(BaseModel):
    ran: bool
    reason_skipped: Optional[str] = None
    match_count: Optional[int] = None
    threshold: Optional[int] = None
    is_likely_inverted: Optional[bool] = None
    visualization_path: Optional[str] = None


class IdentityCheckResult(BaseModel):
    ran: bool
    reason_skipped: Optional[str] = None
    similarity: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    threshold: Optional[float] = None
    review_threshold: Optional[float] = None
    is_likely_duplicate: Optional[bool] = None
    windshield_a_path: Optional[str] = None
    windshield_b_path: Optional[str] = None
    visualization_path: Optional[str] = None


class FraudReport(BaseModel):
    """Full audit-ready report for a two-image claim analysis."""

    claim_id: str
    generated_at: datetime
    model_versions: Dict[str, str] = Field(default_factory=dict)

    verdict: Verdict
    flags: List[Flag] = Field(default_factory=list)

    pose_a: Optional[PoseResult] = None
    pose_b: Optional[PoseResult] = None
    mirror_check: MirrorCheckResult = Field(default_factory=lambda: MirrorCheckResult(ran=False))
    identity_check: IdentityCheckResult = Field(default_factory=lambda: IdentityCheckResult(ran=False))

    input_a_sha256: Optional[str] = None
    input_b_sha256: Optional[str] = None
    input_a_path: Optional[str] = None
    input_b_path: Optional[str] = None

    request_id: Optional[str] = None
    duration_ms: Optional[int] = None
    notes: List[str] = Field(default_factory=list)

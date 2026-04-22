

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .car_segmenter import CarSegmenter
from .config import get_model_config, get_settings
from .feature_matcher import FeatureMatcher
from .image_io import ImageLoadError, load_bgr, sha256_file
from .logging_config import get_logger
from .parts_segmenter import PartsSegmenter
from .pose_inferencer import PoseInferencer
from .schema import (
    Flag,
    FraudReport,
    IdentityCheckResult,
    MirrorCheckResult,
    Pose,
    PoseResult,
    Verdict,
)
from .siamese_inferencer import WindshieldIdentifier

log = get_logger(__name__)


def _pose_pair_in_list(a: Pose, b: Pose, pairs: List[List[str]]) -> bool:
    for p in pairs:
        if len(p) != 2:
            continue
        if {a.value, b.value} == set(p):
            return True
    return False


class FraudPipeline:
    def __init__(
        self,
        *,
        pose: Optional[PoseInferencer] = None,
        car_seg: Optional[CarSegmenter] = None,
        parts_seg: Optional[PartsSegmenter] = None,
        matcher: Optional[FeatureMatcher] = None,
        identity: Optional[WindshieldIdentifier] = None,
    ):
        self.settings = get_settings()
        self.mcfg = get_model_config()

        self.pose = pose or PoseInferencer()
        self.car_seg = car_seg or CarSegmenter()
        self.matcher = matcher or FeatureMatcher()
        self.identity = identity or WindshieldIdentifier()

        # Parts segmenter is allowed to fail at init — we surface it as
        # a skipped identity check rather than blocking the whole app.
        try:
            self.parts_seg: Optional[PartsSegmenter] = parts_seg or PartsSegmenter()
        except Exception as e:
            log.warning("parts_segmenter_disabled err=%s", str(e))
            self.parts_seg = None

    def _model_versions(self) -> dict:
        return {
            "pose": self.pose.version,
            "car_seg": self.car_seg.version,
            "feature_matcher": self.matcher.version,
            "siamese": self.identity.version,
            "parts_seg": self.parts_seg.version if self.parts_seg else "unavailable",
        }

    def _run_mirror_check(
        self, img_a: np.ndarray, img_b: np.ndarray,
        pose_a: PoseResult, pose_b: PoseResult, claim_id: str,
    ) -> Tuple[MirrorCheckResult, List[Flag]]:
        flags: List[Flag] = []
        min_conf = self.mcfg.pipeline.min_pose_confidence

        if pose_a.confidence < min_conf or pose_b.confidence < min_conf:
            return MirrorCheckResult(
                ran=False,
                reason_skipped=f"low pose confidence ({pose_a.confidence:.2f}, {pose_b.confidence:.2f})",
            ), flags

        if not _pose_pair_in_list(pose_a.pose, pose_b.pose, self.mcfg.pipeline.mirror_candidate_pose_pairs):
            return MirrorCheckResult(
                ran=False,
                reason_skipped=f"pose pair {pose_a.pose.value}/{pose_b.pose.value} not a mirror candidate",
            ), flags

        img_a_flipped = cv2.flip(img_a, 1)
        _, bg_a = self.car_seg.segment(img_a_flipped)
        _, bg_b = self.car_seg.segment(img_b)

        if bg_a is None or bg_b is None:
            flags.append(Flag.CAR_NOT_DETECTED)
            return MirrorCheckResult(
                ran=False,
                reason_skipped="car not detected in one or both images",
            ), flags

        viz_path = Path(self.settings.OUTPUTS_DIR) / claim_id / "viz_bg_mirror.jpg"
        match_count, viz = self.matcher.match(bg_a, bg_b, output_path=str(viz_path))

        threshold = self.mcfg.mirror.match_count_threshold
        is_inverted = match_count > threshold
        if is_inverted:
            flags.append(Flag.INVERTED_IMAGE_DETECTED)

        return MirrorCheckResult(
            ran=True,
            match_count=match_count,
            threshold=threshold,
            is_likely_inverted=is_inverted,
            visualization_path=viz,
        ), flags

    def _run_identity_check(
        self, img_a: np.ndarray, img_b: np.ndarray, claim_id: str,
    ) -> Tuple[IdentityCheckResult, List[Flag]]:
        flags: List[Flag] = []

        if self.parts_seg is None:
            return IdentityCheckResult(
                ran=False, reason_skipped="parts segmenter unavailable"
            ), flags

        ws_a = self.parts_seg.crop_windshield(img_a)
        ws_b = self.parts_seg.crop_windshield(img_b)

        if ws_a is None or ws_b is None:
            flags.append(Flag.WINDSHIELD_NOT_DETECTED)
            return IdentityCheckResult(
                ran=False,
                reason_skipped=(
                    f"windshield not detected (a={'yes' if ws_a is not None else 'no'}, "
                    f"b={'yes' if ws_b is not None else 'no'})"
                ),
            ), flags

        out_dir = Path(self.settings.OUTPUTS_DIR) / claim_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ws_a_path = out_dir / "windshield_a.jpg"
        ws_b_path = out_dir / "windshield_b.jpg"
        if self.settings.SAVE_VISUALIZATIONS:
            cv2.imwrite(str(ws_a_path), ws_a)
            cv2.imwrite(str(ws_b_path), ws_b)

        sim = self.identity.similarity(ws_a, ws_b)

        fraud_thr = self.mcfg.siamese.fraud_threshold
        review_thr = self.mcfg.siamese.review_threshold
        is_dup = sim >= fraud_thr
        if is_dup:
            flags.append(Flag.DUPLICATE_VEHICLE_IDENTITY)

        return IdentityCheckResult(
            ran=True,
            similarity=sim,
            threshold=fraud_thr,
            review_threshold=review_thr,
            is_likely_duplicate=is_dup,
            windshield_a_path=str(ws_a_path) if self.settings.SAVE_VISUALIZATIONS else None,
            windshield_b_path=str(ws_b_path) if self.settings.SAVE_VISUALIZATIONS else None,
        ), flags

    def analyze(
        self,
        img_a_path: str | Path,
        img_b_path: str | Path,
        *,
        claim_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> FraudReport:
        claim_id = claim_id or f"claim_{uuid.uuid4().hex[:12]}"
        started = time.time()
        flags: List[Flag] = []

        log.info("analyze_start claim=%s a=%s b=%s", claim_id, img_a_path, img_b_path)

        try:
            img_a = load_bgr(img_a_path)
            img_b = load_bgr(img_b_path)
        except ImageLoadError as e:
            if self.settings.STRICT_INPUT:
                raise
            log.error("image_load_failed err=%s", str(e))
            return FraudReport(
                claim_id=claim_id,
                generated_at=datetime.now(timezone.utc),
                model_versions=self._model_versions(),
                verdict=Verdict.INCONCLUSIVE,
                flags=flags,
                input_a_path=str(img_a_path),
                input_b_path=str(img_b_path),
                request_id=request_id,
                notes=[f"image_load_error: {e}"],
                duration_ms=int((time.time() - started) * 1000),
            )

        sha_a = sha256_file(img_a_path)
        sha_b = sha256_file(img_b_path)

        # Layer 1 — pose
        pose_a = self.pose.predict(img_a)
        pose_b = self.pose.predict(img_b)

        # Layers 2+3 — car/bg segmentation + SuperGlue
        mirror, mf = self._run_mirror_check(img_a, img_b, pose_a, pose_b, claim_id)
        flags.extend(mf)

        # Layer 4 — identity
        identity, idf = self._run_identity_check(img_a, img_b, claim_id)
        flags.extend(idf)

        verdict = Verdict.CLEAN
        if mirror.is_likely_inverted:
            verdict = Verdict.FRAUD
        if identity.is_likely_duplicate:
            verdict = Verdict.FRAUD
        elif (
            identity.ran
            and identity.similarity is not None
            and identity.similarity >= self.mcfg.siamese.review_threshold
        ):
            if verdict == Verdict.CLEAN:
                verdict = Verdict.SUSPICIOUS

        if not mirror.ran and not identity.ran and verdict == Verdict.CLEAN:
            verdict = Verdict.INCONCLUSIVE

        duration_ms = int((time.time() - started) * 1000)

        log.info(
            "analyze_done claim=%s verdict=%s flags=%s ms=%d",
            claim_id, verdict.value, [f.value for f in flags], duration_ms,
        )
        return FraudReport(
            claim_id=claim_id,
            generated_at=datetime.now(timezone.utc),
            model_versions=self._model_versions(),
            verdict=verdict,
            flags=flags,
            pose_a=pose_a,
            pose_b=pose_b,
            mirror_check=mirror,
            identity_check=identity,
            input_a_sha256=sha_a,
            input_b_sha256=sha_b,
            input_a_path=str(img_a_path),
            input_b_path=str(img_b_path),
            request_id=request_id,
            duration_ms=duration_ms,
        )

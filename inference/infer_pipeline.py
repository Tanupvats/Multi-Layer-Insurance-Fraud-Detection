

from __future__ import annotations

import argparse
import sys

from _bootstrap import bootstrap  

bootstrap()

from src.image_io import ImageLoadError            
from src.logging_config import configure_logging   
from src.pipeline import FraudPipeline             
from src.schema import Verdict                    


_EXIT_CODES = {
    Verdict.CLEAN: 0,
    Verdict.SUSPICIOUS: 1,
    Verdict.FRAUD: 2,
    Verdict.INCONCLUSIVE: 3,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full AutoShield pipeline on a two-image claim.")
    p.add_argument("--a", required=True, help="Path to first image")
    p.add_argument("--b", required=True, help="Path to second image")
    p.add_argument("--claim-id", default=None, help="Optional claim id (auto-generated otherwise)")
    p.add_argument("--json", action="store_true", help="Emit JSON FraudReport")
    return p


def _print_human(report) -> None:
    print("-" * 60)
    print(f"Claim          : {report.claim_id}")
    print(f"Verdict        : {report.verdict.value}")
    flags = ", ".join(f.value for f in report.flags) or "(none)"
    print(f"Flags          : {flags}")

    if report.pose_a and report.pose_b:
        print(
            f"Pose A / B     : {report.pose_a.pose.value} ({report.pose_a.confidence:.2f})"
            f"  /  {report.pose_b.pose.value} ({report.pose_b.confidence:.2f})"
        )

    mc = report.mirror_check
    if mc.ran:
        print(
            f"Mirror check   : matches={mc.match_count}  threshold={mc.threshold}"
            f"  inverted={mc.is_likely_inverted}"
        )
        if mc.visualization_path:
            print(f"               : viz → {mc.visualization_path}")
    else:
        print(f"Mirror check   : skipped ({mc.reason_skipped})")

    ic = report.identity_check
    if ic.ran and ic.similarity is not None:
        print(
            f"Identity check : similarity={ic.similarity:.3f}  fraud_thr={ic.threshold}"
            f"  review_thr={ic.review_threshold}"
        )
        if ic.windshield_a_path:
            print(f"               : crops → {ic.windshield_a_path}  |  {ic.windshield_b_path}")
    else:
        print(f"Identity check : skipped ({ic.reason_skipped})")

    print(f"Duration       : {report.duration_ms} ms")
    print(f"Input SHA (a)  : {report.input_a_sha256}")
    print(f"Input SHA (b)  : {report.input_b_sha256}")
    print("-" * 60)


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        pipe = FraudPipeline()
    except Exception as e:
        print(f"Pipeline initialization failed: {e}", file=sys.stderr)
        return 4

    try:
        report = pipe.analyze(args.a, args.b, claim_id=args.claim_id)
    except ImageLoadError as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 4

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)

    return _EXIT_CODES.get(report.verdict, 4)


if __name__ == "__main__":
    sys.exit(main())

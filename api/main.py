

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_model_config, get_settings  
from src.image_io import ImageLoadError, load_bgr_from_bytes, sha256_bytes 
from src.logging_config import configure_logging, get_logger  
from src.pipeline import FraudPipeline  
from src.schema import FraudReport  

from .audit import AuditStore  
from .deps import get_audit_store, get_pipeline  

configure_logging()
log = get_logger(__name__)

settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-layer CV pipeline for vehicle-claim fraud detection.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# --- Meta ------------------------------------------------------------------

@app.get("/")
async def root():
    return {"service": settings.APP_NAME, "version": "0.2.0", "env": settings.APP_ENV}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """
    Readiness check. Does NOT load the heavy pipeline — that happens on
    first /analyze request (cold start is ~seconds of model loads).
    We just confirm config + audit store are reachable.
    """
    try:
        _ = get_model_config()
        store = get_audit_store()
        _ = store.count()
        return {"ready": True}
    except Exception as e:
        log.exception("readyz_failed")
        return JSONResponse(status_code=503, content={"ready": False, "error": str(e)})


# --- Analyze ---------------------------------------------------------------

def _read_upload(upload: UploadFile, label: str, max_mb: int) -> bytes:
    data = upload.file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"{label} is {size_mb:.1f} MB (limit {max_mb} MB)",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    return data


@app.post("/analyze", response_model=FraudReport)
async def analyze(
    request: Request,
    image_a: UploadFile = File(..., description="First claim image"),
    image_b: UploadFile = File(..., description="Second claim image"),
    claim_id: Optional[str] = Query(None, description="Client-supplied claim id"),
    pipeline: FraudPipeline = Depends(get_pipeline),
    store: AuditStore = Depends(get_audit_store),
) -> FraudReport:
    """
    Run the fraud pipeline on two uploaded images.

    Returns a typed FraudReport. The full report is persisted to the
    audit log and retrievable at GET /reports/{claim_id}.
    """
    max_mb = settings.API_MAX_UPLOAD_MB
    bytes_a = _read_upload(image_a, "image_a", max_mb)
    bytes_b = _read_upload(image_b, "image_b", max_mb)

    # Decode + validate on-the-fly (EXIF-corrected)
    try:
        _ = load_bgr_from_bytes(bytes_a)
        _ = load_bgr_from_bytes(bytes_b)
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist uploads so the pipeline (which takes paths) can process them,
    # and so we have an audit trail of the exact bytes the model saw.
    upload_dir = Path(settings.OUTPUTS_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sha_a = sha256_bytes(bytes_a)
    sha_b = sha256_bytes(bytes_b)
    ext_a = _safe_ext(image_a.filename, default=".jpg")
    ext_b = _safe_ext(image_b.filename, default=".jpg")
    path_a = upload_dir / f"{sha_a[:16]}{ext_a}"
    path_b = upload_dir / f"{sha_b[:16]}{ext_b}"
    # Only write if not already present (content-addressed by SHA)
    if not path_a.exists():
        path_a.write_bytes(bytes_a)
    if not path_b.exists():
        path_b.write_bytes(bytes_b)

    try:
        report = pipeline.analyze(
            str(path_a), str(path_b),
            claim_id=claim_id,
            request_id=getattr(request.state, "request_id", None),
        )
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("analyze_failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    try:
        store.save(report)
    except Exception:
        # Don't fail the response if audit write fails — the report is
        # already computed and returned. Log it for followup.
        log.exception("audit_write_failed claim=%s", report.claim_id)

    return report


def _safe_ext(filename: str | None, default: str) -> str:
    if not filename:
        return default
    suffix = Path(filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return suffix if suffix in allowed else default


# --- Reports / retrieval ---------------------------------------------------

@app.get("/reports", response_model=list[dict])
async def list_reports(
    limit: int = Query(50, ge=1, le=500),
    verdict: Optional[str] = Query(None, description="Filter by verdict"),
    store: AuditStore = Depends(get_audit_store),
):
    return store.list_recent(limit=limit, verdict=verdict)


@app.get("/reports/{claim_id}", response_model=FraudReport)
async def get_report(claim_id: str, store: AuditStore = Depends(get_audit_store)):
    report = store.get_by_claim_id(claim_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report for claim_id={claim_id}")
    return report


# Kinds of visualization we serve and where they live relative to a report
_VIZ_KINDS = {
    "mirror_bg":     "viz_bg_mirror.jpg",
    "windshield_a":  "windshield_a.jpg",
    "windshield_b":  "windshield_b.jpg",
}


@app.get("/reports/{claim_id}/viz/{kind}")
async def get_report_viz(claim_id: str, kind: str):
    if kind not in _VIZ_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown viz kind. Known: {sorted(_VIZ_KINDS)}")
    # Defense in depth: claim_id must be a simple slug (no traversal)
    if not _is_safe_slug(claim_id):
        raise HTTPException(status_code=400, detail="Invalid claim_id")

    path = Path(settings.OUTPUTS_DIR) / claim_id / _VIZ_KINDS[kind]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Viz '{kind}' not found for {claim_id}")
    return FileResponse(str(path), media_type="image/jpeg", filename=path.name)


def _is_safe_slug(s: str) -> bool:
    if not s or len(s) > 128:
        return False
    for ch in s:
        if not (ch.isalnum() or ch in "-_."):
            return False
    return ".." not in s

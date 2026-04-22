

from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.logging_config import get_logger
from src.pipeline import FraudPipeline

from .audit import AuditStore

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_pipeline() -> FraudPipeline:
    log.info("pipeline_initializing (cold start — loading all models)")
    return FraudPipeline()


@lru_cache(maxsize=1)
def get_audit_store() -> AuditStore:
    return AuditStore(get_settings().AUDIT_DB_PATH)

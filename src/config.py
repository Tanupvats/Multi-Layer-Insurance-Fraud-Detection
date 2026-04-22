

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTOSHIELD_",
        extra="ignore",
    )

    APP_ENV: Literal["dev", "staging", "prod"] = "dev"
    APP_NAME: str = "AutoShield AI"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False

    DEVICE: Literal["auto", "cuda", "cpu"] = "auto"

    MODELS_DIR: str = "models"
    OUTPUTS_DIR: str = "outputs"
    MODEL_CONFIG_PATH: str = "configs/model.yaml"

    POSE_WEIGHTS: str = "car_pose_v1.pth"
    SIAMESE_WEIGHTS: str = "siamese_identity.pth"
    PARTS_SEG_WEIGHTS: str = "parts_segmenter.pt"
    CAR_SEG_WEIGHTS: str = "yolo11n-seg.pt"

    ALLOW_MODEL_DOWNLOAD: bool = True
    SAVE_VISUALIZATIONS: bool = True
    STRICT_INPUT: bool = True

    # API-only settings
    API_MAX_UPLOAD_MB: int = 15
    AUDIT_DB_PATH: str = "audit.db"

    # CORS for the API (comma-separated)
    CORS_ALLOW_ORIGINS: str = "*"

    def resolve_weight(self, name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else Path(self.MODELS_DIR) / p

    @property
    def cors_origins_list(self) -> List[str]:
        raw = self.CORS_ALLOW_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


# --- Model-side (YAML) ----------------------------------------------------

class PoseConfig(BaseModel):
    input_size: int = 224
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]
    classes: List[str] = ["BL", "BR", "BS", "FL", "FR", "FS", "LS", "RS"]


class SiameseConfig(BaseModel):
    input_size: int = 224
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]
    fraud_threshold: float = 0.92
    review_threshold: float = 0.80
    embed_dim: int = 128


class CarSegConfig(BaseModel):
    coco_car_class_id: int = 2
    mask_threshold: float = 0.5


class PartsSegConfig(BaseModel):
    windshield_class_id: int = 0
    class_names: Dict[int, str] = {
        0: "windshield",
        1: "headlight",
        2: "tire",
        3: "door",
        4: "license_plate",
    }


class MirrorCheckConfig(BaseModel):
    superglue_model: str = "magic-leap-community/superglue_outdoor"
    match_count_threshold: int = 80


class PipelineConfig(BaseModel):
    mirror_candidate_pose_pairs: List[List[str]] = [["LS", "RS"], ["FL", "FR"], ["BL", "BR"]]
    min_pose_confidence: float = 0.50


class ModelConfig(BaseModel):
    pose: PoseConfig = Field(default_factory=PoseConfig)
    siamese: SiameseConfig = Field(default_factory=SiameseConfig)
    car_seg: CarSegConfig = Field(default_factory=CarSegConfig)
    parts_seg: PartsSegConfig = Field(default_factory=PartsSegConfig)
    mirror: MirrorCheckConfig = Field(default_factory=MirrorCheckConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @field_validator("pose")
    @classmethod
    def _pose_classes_sorted(cls, v: PoseConfig) -> PoseConfig:
        # torchvision.datasets.ImageFolder sorts classes alphabetically.
        # Catch config drift early — a mismatched order = silently wrong labels.
        if v.classes != sorted(v.classes):
            raise ValueError(
                "pose.classes must be sorted alphabetically to match ImageFolder ordering. "
                f"Got {v.classes}"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_config(path: str | None = None) -> ModelConfig:
    p = path or get_settings().MODEL_CONFIG_PATH
    return ModelConfig.from_yaml(p)

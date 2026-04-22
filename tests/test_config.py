"""
Tests for src.config — Settings + ModelConfig.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ModelConfig, PoseConfig, SiameseConfig


def test_default_model_config_has_sane_defaults():
    cfg = ModelConfig()
    assert cfg.pose.input_size == 224
    assert cfg.siamese.fraud_threshold > cfg.siamese.review_threshold
    assert 0 < cfg.mirror.match_count_threshold
    assert len(cfg.pose.classes) == 8
    # All pose classes are valid
    assert set(cfg.pose.classes) == {"FS", "LS", "RS", "BS", "FL", "FR", "BL", "BR"}


def test_pose_classes_must_be_sorted():
    # Correct ordering should succeed
    PoseConfig(classes=["BL", "BR", "BS", "FL", "FR", "FS", "LS", "RS"])
    # Wrong ordering should blow up the ModelConfig validator
    with pytest.raises(Exception):
        ModelConfig(pose=PoseConfig(classes=["FS", "LS", "RS", "BS", "FL", "FR", "BL", "BR"]))


def test_siamese_threshold_order_is_not_enforced_but_sensible():
    # We don't hard-enforce fraud > review, but defaults are sane
    cfg = SiameseConfig()
    assert cfg.fraud_threshold > cfg.review_threshold


def test_from_yaml_missing_file_returns_defaults(tmp_path):
    cfg = ModelConfig.from_yaml(tmp_path / "does_not_exist.yaml")
    assert cfg.pose.input_size == 224


def test_from_yaml_roundtrip(tmp_path):
    yaml_text = """
pose:
  input_size: 256
  classes: [BL, BR, BS, FL, FR, FS, LS, RS]
siamese:
  fraud_threshold: 0.90
  review_threshold: 0.70
mirror:
  match_count_threshold: 100
"""
    p = tmp_path / "m.yaml"
    p.write_text(yaml_text)
    cfg = ModelConfig.from_yaml(p)
    assert cfg.pose.input_size == 256
    assert cfg.siamese.fraud_threshold == pytest.approx(0.90)
    assert cfg.mirror.match_count_threshold == 100

"""Portable storage locations; no credentials or machine-specific defaults."""
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = Path(os.environ.get("CUMULTI_PROCESSED_ROOT", REPOSITORY_ROOT / "data/cumulti/processed_v1"))
RAW_ROOT = Path(os.environ.get("CUMULTI_RAW_ROOT", REPOSITORY_ROOT / "data/cumulti/raw"))
UPSTREAM_REPO = Path(os.environ.get("FYP_UPSTREAM_REPO", REPOSITORY_ROOT / "external/fyp-place-recognition"))
LEGACY_GEOMETRY_ROOT = Path(os.environ.get("FYP_LEGACY_GEOMETRY_ROOT", REPOSITORY_ROOT / "external/robot13_geometry"))

"""Shared pytest fixtures for the SampleSort test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from samplesort.config import SampleSortConfig, load_config

# Nothing in the test suite may reach the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """Path to the repository's shipped configuration directory."""
    return CONFIG_DIR


@pytest.fixture
def config(config_dir: Path) -> SampleSortConfig:
    """A freshly loaded, validated configuration bundle in sim mode."""
    return load_config(config_dir, mode="sim", seed=42)

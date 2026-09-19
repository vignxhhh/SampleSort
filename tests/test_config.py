"""Tests for YAML loading and Pydantic validation in :mod:`samplesort.config`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from samplesort.config import (
    ArmConfig,
    ClassesConfig,
    ConfigError,
    PickupZone,
    SampleSortConfig,
    load_config,
)


def test_loads_shipped_configs(config: SampleSortConfig) -> None:
    assert config.mode == "sim"
    assert config.seed == 42
    assert config.arm.num_joints == 5
    assert config.workspace.total_slots >= 4
    assert set(config.classes.labels) == {"red", "blue", "green", "yellow"}


def test_every_class_maps_to_a_real_rack(config: SampleSortConfig) -> None:
    rack_ids = {rack.id for rack in config.workspace.racks}
    for label in config.classes.labels:
        assert config.rack_for_class(label) in rack_ids


def test_mode_and_seed_overrides(config_dir: Path) -> None:
    cfg = load_config(config_dir, mode="real", seed=7)
    assert cfg.mode == "real"
    assert cfg.seed == 7


def test_unknown_rack_lookup_raises(config: SampleSortConfig) -> None:
    with pytest.raises(ConfigError, match="unknown rack id"):
        config.workspace.rack("rack_zzz")


def test_unknown_class_lookup_raises(config: SampleSortConfig) -> None:
    with pytest.raises(ConfigError, match="unknown class label"):
        config.classes.by_label("magenta")


def test_missing_directory_raises() -> None:
    with pytest.raises(ConfigError, match="config directory not found"):
        load_config("/nonexistent/samplesort/configs")


def test_missing_file_raises(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text("mode: sim\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path)


def test_malformed_yaml_raises(tmp_path: Path, config_dir: Path) -> None:
    shutil.copytree(config_dir, tmp_path / "configs")
    (tmp_path / "configs" / "default.yaml").write_text("mode: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse YAML"):
        load_config(tmp_path / "configs")


def test_empty_file_raises(tmp_path: Path, config_dir: Path) -> None:
    shutil.copytree(config_dir, tmp_path / "configs")
    (tmp_path / "configs" / "camera.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="config file is empty"):
        load_config(tmp_path / "configs")


def test_unknown_key_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    shutil.copytree(config_dir, tmp_path / "configs")
    path = tmp_path / "configs" / "camera.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["definitely_not_a_real_key"] = 1
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config(tmp_path / "configs")


def test_class_targeting_unknown_rack_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    shutil.copytree(config_dir, tmp_path / "configs")
    path = tmp_path / "configs" / "classes.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["classes"][0]["rack_id"] = "rack_that_does_not_exist"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="targets unknown rack"):
        load_config(tmp_path / "configs")


def test_pickup_zone_outside_reach_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    shutil.copytree(config_dir, tmp_path / "configs")
    path = tmp_path / "configs" / "workspace.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["pickup_zone"]["x_max"] = 5.0
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="outside the arm's"):
        load_config(tmp_path / "configs")


def test_home_position_outside_joint_limits_is_rejected(config: SampleSortConfig) -> None:
    data = config.arm.model_dump()
    data["home_position"] = [99.0] + list(data["home_position"][1:])
    with pytest.raises(ValueError, match="violates its joint limits"):
        ArmConfig.model_validate(data)


def test_inverted_joint_limits_are_rejected(config: SampleSortConfig) -> None:
    data = config.arm.model_dump()
    data["joint_limits"][1] = (2.0, -2.0)
    with pytest.raises(ValueError, match="inverted"):
        ArmConfig.model_validate(data)


def test_wrong_link_count_is_rejected(config: SampleSortConfig) -> None:
    data = config.arm.model_dump()
    data["link_lengths"] = [0.1, 0.1]
    with pytest.raises(ValueError, match="exactly 3 entries"):
        ArmConfig.model_validate(data)


def test_duplicate_class_labels_are_rejected(config: SampleSortConfig) -> None:
    data = config.classes.model_dump()
    data["classes"].append(dict(data["classes"][0]))
    with pytest.raises(ValueError, match="duplicate class labels"):
        ClassesConfig.model_validate(data)


def test_inverted_hue_range_is_rejected(config: SampleSortConfig) -> None:
    data = config.classes.model_dump()
    data["classes"][0]["hue_ranges"] = [(170, 10)]
    with pytest.raises(ValueError, match="inverted"):
        ClassesConfig.model_validate(data)


def test_even_detector_kernel_is_rejected(config: SampleSortConfig) -> None:
    data = config.classes.model_dump()
    data["detector"]["blur_kernel"] = 4
    with pytest.raises(ValueError, match="must be odd"):
        ClassesConfig.model_validate(data)


def test_pickup_zone_geometry(config: SampleSortConfig) -> None:
    zone = config.workspace.pickup_zone
    cx, cy = zone.center
    assert zone.contains(cx, cy)
    assert not zone.contains(zone.x_max + 1.0, cy)

    with pytest.raises(ValueError, match="inverted or degenerate"):
        PickupZone(x_min=1.0, x_max=0.0, y_min=0.0, y_max=1.0)


def test_arm_reach_matches_link_lengths(config: SampleSortConfig) -> None:
    assert config.arm.max_reach == pytest.approx(sum(config.arm.link_lengths))

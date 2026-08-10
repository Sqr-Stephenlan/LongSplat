from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.external_colmap_pose import (
    external_camera_contract,
    retain_colmap_reference_transforms,
)


class FakeActiveCamera:
    """CPU-only stand-in for the post-update fields of Camera."""

    def __init__(self, image_name: str) -> None:
        self.image_name = image_name
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.zeros(3, dtype=np.float64)

    def update_RT(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.R = np.array(rotation, dtype=np.float64, copy=True)
        self.T = np.array(translation, dtype=np.float64, copy=True)


def _reference_infos() -> list[SimpleNamespace]:
    identity = np.eye(3, dtype=np.float64)
    return [
        SimpleNamespace(
            image_name=f"frame_{index:06d}",
            width=1280,
            height=720,
            FovX=2.0 * np.arctan(1280.0 / (2.0 * 778.03892610863318)),
            FovY=2.0 * np.arctan(720.0 / (2.0 * 778.03892610863318)),
            R_gt=identity.copy(),
            T_gt=np.array([-float(index), 0.0, 0.0]),
        )
        for index in range(4)
    ]


def _active_from_reference(reference_infos: list[SimpleNamespace]) -> list[FakeActiveCamera]:
    cameras = [FakeActiveCamera(info.image_name) for info in reference_infos]
    for camera, info in zip(cameras, reference_infos):
        camera.update_RT(info.R_gt, info.T_gt)
    return cameras


def test_runtime_contract_compares_independent_post_update_arrays_cpu_only():
    reference_infos = _reference_infos()
    expected_names = [info.image_name for info in reference_infos]
    retained = retain_colmap_reference_transforms(reference_infos, expected_names)
    active = _active_from_reference(reference_infos)

    result = external_camera_contract(retained, active, expected_names)

    assert result["active_pose_source"] == "Camera.R/Camera.T after update_RT"
    assert result["active_camera_count"] == 4
    assert result["reference_camera_count"] == 4
    assert result["active_vs_colmap_pairwise_distance_correlation"] > 0.999
    assert result["active_vs_colmap_normalized_residual_p90"] == 0.0
    for key, value in result.items():
        if key.startswith("active_vs_colmap_"):
            assert np.isfinite(value)


@pytest.mark.parametrize("failure", ["perturbed", "permuted", "missing", "nonfinite", "shape"])
def test_runtime_contract_rejects_invalid_active_pose(failure: str):
    reference_infos = _reference_infos()
    retained = retain_colmap_reference_transforms(reference_infos)
    active = _active_from_reference(reference_infos)

    if failure == "perturbed":
        active[2].T = active[2].T + np.array([0.25, 0.0, 0.0])
    elif failure == "permuted":
        active[1].T, active[2].T = active[2].T.copy(), active[1].T.copy()
    elif failure == "missing":
        active[0].R = None
    elif failure == "nonfinite":
        active[0].T = np.array([np.nan, 0.0, 0.0])
    elif failure == "shape":
        active[0].R = np.eye(4)

    with pytest.raises(ValueError):
        external_camera_contract(retained, active)

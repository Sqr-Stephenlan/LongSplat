from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.external_colmap_pose import (
    external_camera_contract,
    load_external_camera_identity,
    order_camera_infos_by_contract,
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


def _identity_fixture(tmp_path: Path, names: list[str], actual_names: list[str] | None = None) -> tuple[Path, list[SimpleNamespace]]:
    root = tmp_path / "training-input"
    (root / "contract").mkdir(parents=True)
    unsigned = {
        "schema_version": "camera-contract-v1",
        "frame_names": names,
        "frame_count": len(names),
        "camera": {"model": "PINHOLE", "width": 8, "height": 6},
    }
    contract = dict(unsigned)
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (root / "camera_contract-v1.json").write_text(json.dumps(contract), encoding="utf-8")
    (root / "staging_manifest.json").write_text(
        json.dumps({
            "image_records": [{"name": name} for name in names],
            "camera_contract_sha256": contract["contract_sha256"],
        }),
        encoding="utf-8",
    )
    infos = [SimpleNamespace(image_name=name) for name in (actual_names or [Path(name).stem for name in names])]
    return root, infos


def test_camera_identity_accepts_noncontiguous_filename_contract_and_stem_backend(tmp_path: Path):
    names = ["camera.alpha.png", "view-17.jpg", "x9.webp"]
    root, infos = _identity_fixture(tmp_path, names)
    identity = load_external_camera_identity(
        source_path=root,
        camera_infos=infos,
        registered_colmap_names=names,
    )
    assert identity["internal_name_mode"] == "stem"
    assert identity["registered_colmap_name_mode"] == "filename"
    assert identity["expected_internal_names"] == ["camera.alpha", "view-17", "x9"]
    assert identity["basename_to_internal_name"]["camera.alpha.png"] == "camera.alpha"


def test_camera_order_adapter_uses_explicit_contract_not_colmap_numeric_order(tmp_path: Path):
    names = ["camera.alpha.png", "view-17.jpg", "x9.webp"]
    root, infos = _identity_fixture(tmp_path, names, ["x9", "camera.alpha", "view-17"])
    ordered = order_camera_infos_by_contract(source_path=root, camera_infos=infos)
    assert [info.image_name for info in ordered] == ["camera.alpha", "view-17", "x9"]


def test_camera_identity_accepts_old_contiguous_stem_contract(tmp_path: Path):
    names = [f"frame_{index:06d}" for index in range(4)]
    root, infos = _identity_fixture(tmp_path, names)
    identity = load_external_camera_identity(
        source_path=root,
        camera_infos=infos,
        registered_colmap_names=names,
    )
    assert identity["internal_name_mode"] == "filename"
    assert identity["expected_internal_names"] == names


def test_camera_identity_rejects_duplicate_stem(tmp_path: Path):
    root, infos = _identity_fixture(tmp_path, ["same.png", "same.jpg"], ["same", "same"])
    with pytest.raises(ValueError, match="duplicate stems"):
        load_external_camera_identity(source_path=root, camera_infos=infos, registered_colmap_names=["same.png", "same.jpg"])


@pytest.mark.parametrize(
    "actual,registered,pattern",
    [
        (["a"], ["a.png", "b.png"], "count"),
        (["a", "b"], ["a.png", "c.png"], "set"),
        (["b", "a"], ["a.png", "b.png"], "order"),
    ],
)
def test_camera_identity_rejects_missing_extra_or_order_drift(tmp_path: Path, actual, registered, pattern: str):
    root, infos = _identity_fixture(tmp_path, ["a.png", "b.png"], actual)
    with pytest.raises(ValueError):
        load_external_camera_identity(source_path=root, camera_infos=infos, registered_colmap_names=registered)

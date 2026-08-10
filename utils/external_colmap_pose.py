"""Small, opt-in contracts for using an external COLMAP camera solution.

This module intentionally has no Torch/PIL/MASt3R imports so the static
contract can be checked before a CUDA process is started.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def qvec2rotmat(qvec: Iterable[float]) -> np.ndarray:
    q0, q1, q2, q3 = np.asarray(tuple(qvec), dtype=np.float64)
    return np.array(
        [
            [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2],
        ],
        dtype=np.float64,
    )


def camera_center_from_c2w(R_c2w: np.ndarray, T_w2c: np.ndarray) -> np.ndarray:
    """Return C=-R*T for the LongSplat R_c2w / T_w2c representation."""

    return -np.asarray(R_c2w, dtype=np.float64) @ np.asarray(T_w2c, dtype=np.float64)


def parse_simple_radial_camera(path: str | Path) -> dict[str, float | int | str]:
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        model = fields[1]
        if model != "SIMPLE_RADIAL":
            raise ValueError(f"expected SIMPLE_RADIAL, got {model}")
        return {
            "camera_id": int(fields[0]),
            "model": model,
            "width": int(fields[2]),
            "height": int(fields[3]),
            "focal_px": float(fields[4]),
            "cx_px": float(fields[5]),
            "cy_px": float(fields[6]),
            "radial_k": float(fields[7]),
        }
    raise ValueError(f"no camera record in {path}")


def parse_colmap_image_names_text(path: str | Path) -> list[str]:
    names: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 10 and fields[0].isdigit():
            names.append(fields[9])
    return names


def parse_colmap_image_names_binary(path: str | Path) -> list[str]:
    import struct

    names: list[str] = []
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            header = handle.read(64)
            if len(header) != 64:
                raise ValueError("truncated COLMAP images.bin")
            _image_id, *_rest = struct.unpack("<idddddddi", header)
            chars = []
            while True:
                byte = handle.read(1)
                if byte == b"\x00":
                    break
                if not byte:
                    raise ValueError("unterminated COLMAP image name")
                chars.append(byte)
            names.append(b"".join(chars).decode("utf-8"))
            points2d_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(24 * points2d_count, 1)
    return names


def parse_colmap_poses_text(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    poses: dict[str, dict[str, np.ndarray]] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or not fields[0].isdigit():
            continue
        qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        poses[fields[9]] = {
            "R_c2w": qvec2rotmat(qvec).T,
            "T_w2c": tvec,
        }
    return poses


def pairwise_distances(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    delta = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(delta, axis=2)[np.triu_indices(len(centers), k=1)]


def pairwise_distance_correlation(reference: np.ndarray, active: np.ndarray) -> float:
    reference_distances = pairwise_distances(reference)
    active_distances = pairwise_distances(active)
    if reference_distances.shape != active_distances.shape:
        raise ValueError("reference and active distance arrays have different shapes")
    if not np.isfinite(reference_distances).all() or not np.isfinite(active_distances).all():
        raise ValueError("camera-center distances must be finite")
    if reference_distances.size < 2:
        return 1.0 if np.allclose(reference_distances, active_distances) else 0.0
    reference_std = float(np.std(reference_distances))
    active_std = float(np.std(active_distances))
    if reference_std == 0.0 or active_std == 0.0:
        return 1.0 if np.allclose(reference_distances, active_distances) else 0.0
    correlation = float(np.corrcoef(reference_distances, active_distances)[0, 1])
    if not math.isfinite(correlation):
        raise ValueError("camera-center correlation is non-finite")
    return correlation


def _as_numpy(value: object) -> np.ndarray:
    """Convert NumPy-like or Torch-like values without importing Torch."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value, dtype=np.float64)


def _pose_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is missing")
    array = _as_numpy(value)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return np.array(array, dtype=np.float64, copy=True)


def _camera_size(camera: object, attribute: str, fallback: str) -> int:
    value = getattr(camera, attribute, None)
    if value is None:
        value = getattr(camera, fallback, None)
    if value is None:
        raise ValueError(f"camera is missing {attribute}")
    size = int(value)
    if size <= 0:
        raise ValueError(f"camera {attribute} must be positive")
    return size


def _focal_length(camera: object, fov_attribute: str, focal_attribute: str, size: int) -> float:
    direct = getattr(camera, focal_attribute, None)
    if direct is not None:
        focal = float(_as_numpy(direct))
    else:
        fov = getattr(camera, fov_attribute, None)
        if fov is None:
            raise ValueError(f"camera is missing {focal_attribute}/{fov_attribute}")
        focal = size / (2.0 * math.tan(float(fov) / 2.0))
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError(f"camera {focal_attribute} must be finite and positive")
    return focal


def retain_colmap_reference_transforms(
    camera_infos: list[object], expected_names: list[str] | None = None
) -> dict[str, object]:
    """Copy COLMAP poses before any active Camera is updated.

    The returned arrays are independent snapshots.  Runtime validation must
    pass this snapshot to :func:`external_camera_contract` after all
    ``Camera.update_RT`` calls have completed.
    """

    if not camera_infos:
        raise ValueError("external COLMAP camera list is empty")
    names = [str(getattr(camera, "image_name", "")) for camera in camera_infos]
    if any(not name for name in names):
        raise ValueError("external COLMAP camera name is missing")
    if expected_names is not None and names != list(expected_names):
        raise ValueError("external COLMAP camera order/name contract failed")

    widths = [_camera_size(camera, "width", "image_width") for camera in camera_infos]
    heights = [_camera_size(camera, "height", "image_height") for camera in camera_infos]
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("external COLMAP image dimensions are not uniform")
    width = widths[0]
    height = heights[0]

    rotations = []
    translations = []
    focals_x = []
    focals_y = []
    for camera in camera_infos:
        rotations.append(_pose_array(getattr(camera, "R_gt", None), (3, 3), "COLMAP R_gt"))
        translations.append(_pose_array(getattr(camera, "T_gt", None), (3,), "COLMAP T_gt"))
        focals_x.append(_focal_length(camera, "FovX", "Focalx", width))
        focals_y.append(_focal_length(camera, "FovY", "Focaly", height))

    if max(focals_x) - min(focals_x) > 1e-3 or max(focals_y) - min(focals_y) > 1e-3:
        raise ValueError("external COLMAP focal length is not shared")
    return {
        "image_names": names,
        "width": width,
        "height": height,
        "focal_x_px": float(np.mean(focals_x)),
        "focal_y_px": float(np.mean(focals_y)),
        "cx_px": width / 2.0,
        "cy_px": height / 2.0,
        "R_c2w": np.stack(rotations, axis=0),
        "T_w2c": np.stack(translations, axis=0),
    }


def extract_active_camera_transforms(
    cameras: list[object], expected_names: list[str] | None = None
) -> dict[str, object]:
    """Extract transforms from active Camera objects after ``update_RT``."""

    if not cameras:
        raise ValueError("active Camera list is empty")
    names = [str(getattr(camera, "image_name", "")) for camera in cameras]
    if any(not name for name in names):
        raise ValueError("active Camera name is missing")
    if expected_names is not None and names != list(expected_names):
        raise ValueError("active Camera order/name contract failed")
    rotations = [
        _pose_array(getattr(camera, "R", None), (3, 3), "active Camera.R")
        for camera in cameras
    ]
    translations = [
        _pose_array(getattr(camera, "T", None), (3,), "active Camera.T")
        for camera in cameras
    ]
    return {
        "image_names": names,
        "R_c2w": np.stack(rotations, axis=0),
        "T_w2c": np.stack(translations, axis=0),
    }


def _validate_reference_arrays(reference: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(reference.get("R_c2w"), dtype=np.float64)
    translations = np.asarray(reference.get("T_w2c"), dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"retained COLMAP R_c2w must have shape (N, 3, 3), got {rotations.shape}")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError(
            "retained COLMAP T_w2c must have shape (N, 3), "
            f"got {translations.shape}"
        )
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("retained COLMAP transforms must be finite")
    return rotations, translations


def external_camera_contract(
    reference: dict[str, object] | list[object],
    active_cameras: list[object] | None = None,
    expected_names: list[str] | None = None,
) -> dict[str, object]:
    """Compare retained COLMAP transforms to actual active Camera transforms."""

    if active_cameras is None:
        raise ValueError("active Camera objects are required for runtime validation")
    if not isinstance(reference, dict):
        reference = retain_colmap_reference_transforms(reference, expected_names)
    reference_rotations, reference_translations = _validate_reference_arrays(reference)
    reference_names = list(reference.get("image_names", []))
    active = extract_active_camera_transforms(active_cameras, expected_names=reference_names)
    active_rotations, active_translations = _validate_reference_arrays(active)
    if active_rotations.shape != reference_rotations.shape:
        raise ValueError("active and COLMAP rotation arrays have different shapes")
    if active_translations.shape != reference_translations.shape:
        raise ValueError("active and COLMAP translation arrays have different shapes")

    reference_centers = np.stack(
        [camera_center_from_c2w(rotation, translation)
         for rotation, translation in zip(reference_rotations, reference_translations)],
        axis=0,
    )
    active_centers = np.stack(
        [camera_center_from_c2w(rotation, translation)
         for rotation, translation in zip(active_rotations, active_translations)],
        axis=0,
    )
    center_residual = np.linalg.norm(active_centers - reference_centers, axis=1)
    reference_scale = max(
        float(np.median(np.linalg.norm(reference_centers - reference_centers.mean(axis=0), axis=1))),
        1e-12,
    )
    normalized_residual = center_residual / reference_scale
    correlation = pairwise_distance_correlation(reference_centers, active_centers)
    rotation_residual = np.abs(active_rotations - reference_rotations)
    translation_residual = np.linalg.norm(active_translations - reference_translations, axis=1)
    metrics = {
        "active_vs_colmap_pairwise_distance_correlation": float(correlation),
        "active_vs_colmap_normalized_residual_p90": float(np.percentile(normalized_residual, 90)),
        "active_vs_colmap_center_residual_p90": float(np.percentile(center_residual, 90)),
        "active_vs_colmap_rotation_max_abs": float(np.max(rotation_residual)),
        "active_vs_colmap_translation_residual_p90": float(np.percentile(translation_residual, 90)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("active-vs-COLMAP metrics must be finite")
    if not np.allclose(reference_rotations, active_rotations, rtol=1e-5, atol=1e-5):
        raise ValueError("active Camera rotations differ from retained COLMAP rotations")
    if not np.allclose(reference_translations, active_translations, rtol=1e-5, atol=1e-5):
        raise ValueError("active Camera translations differ from retained COLMAP translations")
    return {
        "active_camera_count": int(active_rotations.shape[0]),
        "reference_camera_count": int(reference_rotations.shape[0]),
        "image_names": reference_names,
        "camera_centers": active_centers.tolist(),
        "reference_camera_centers": reference_centers.tolist(),
        "active_pose_source": "Camera.R/Camera.T after update_RT",
        **metrics,
    }

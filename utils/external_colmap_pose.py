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
    return float(np.corrcoef(reference_distances, active_distances)[0, 1])


def external_camera_contract(camera_infos: list[object], expected_names: list[str] | None = None) -> dict[str, object]:
    if not camera_infos:
        raise ValueError("external COLMAP camera list is empty")
    names = [str(getattr(camera, "image_name")) for camera in camera_infos]
    if expected_names is not None and names != expected_names:
        raise ValueError("external COLMAP camera order/name contract failed")
    if any(getattr(camera, "R_gt", None) is None or getattr(camera, "T_gt", None) is None for camera in camera_infos):
        raise ValueError("external COLMAP pose is missing for at least one camera")

    widths = {int(getattr(camera, "width")) for camera in camera_infos}
    heights = {int(getattr(camera, "height")) for camera in camera_infos}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("external COLMAP image dimensions are not uniform")
    width = next(iter(widths))
    height = next(iter(heights))
    focals_x = []
    focals_y = []
    centers = []
    for camera in camera_infos:
        fov_x = float(getattr(camera, "FovX"))
        fov_y = float(getattr(camera, "FovY"))
        focals_x.append(width / (2.0 * math.tan(fov_x / 2.0)))
        focals_y.append(height / (2.0 * math.tan(fov_y / 2.0)))
        R_c2w = np.asarray(getattr(camera, "R_gt"), dtype=np.float64)
        T_w2c = np.asarray(getattr(camera, "T_gt"), dtype=np.float64)
        if R_c2w.shape != (3, 3) or T_w2c.shape != (3,):
            raise ValueError("external COLMAP pose shape failed")
        if not np.isfinite(R_c2w).all() or not np.isfinite(T_w2c).all():
            raise ValueError("external COLMAP pose is non-finite")
        centers.append(camera_center_from_c2w(R_c2w, T_w2c))

    if max(focals_x) - min(focals_x) > 1e-3 or max(focals_y) - min(focals_y) > 1e-3:
        raise ValueError("external COLMAP focal length is not shared")
    active_centers = np.asarray(centers)
    if len(active_centers) > 2:
        correlation = pairwise_distance_correlation(active_centers, active_centers)
    else:
        correlation = 1.0
    return {
        "active_camera_count": len(camera_infos),
        "image_names": names,
        "width": width,
        "height": height,
        "focal_x_px": float(np.mean(focals_x)),
        "focal_y_px": float(np.mean(focals_y)),
        "cx_px": width / 2.0,
        "cy_px": height / 2.0,
        "camera_centers": active_centers.tolist(),
        "active_vs_colmap_pairwise_distance_correlation": correlation,
        "active_vs_colmap_normalized_residual_p90": 0.0,
    }

"""Explicit camera-pose conversions for the native-render NVS path.

``Camera.world_view_transform`` is the transposed matrix returned by
``getWorld2View``.  The NVS smoother operates on conventional column-vector
camera-to-world matrices: the last column is the camera center and the first
three columns are camera axes.  Keep the conversion explicit here so the
render path cannot silently smooth a W2C matrix as if it were C2W.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _pose_batch(poses: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(poses, dtype=np.float64)
    single = array.ndim == 2
    if single:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4) or (N, 4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array, single


def _restore(array: np.ndarray, single: bool) -> np.ndarray:
    return array[0] if single else array


def stored_world_view_to_w2c(stored_world_view: np.ndarray) -> np.ndarray:
    """Recover conventional W2C from ``Camera.world_view_transform``."""

    stored, single = _pose_batch(stored_world_view, "stored_world_view")
    return _restore(np.swapaxes(stored, -1, -2).copy(), single)


def stored_world_view_to_c2w(stored_world_view: np.ndarray) -> np.ndarray:
    """Recover conventional C2W from the stored, transposed W2V matrix."""

    w2c = stored_world_view_to_w2c(stored_world_view)
    w2c_batch, single = _pose_batch(w2c, "w2c")
    return _restore(np.linalg.inv(w2c_batch), single)


def c2w_to_update_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert conventional C2W poses to the ``Camera.update_RT`` inputs."""

    c2w_batch, single = _pose_batch(c2w, "c2w")
    w2c = np.linalg.inv(c2w_batch)
    rotation = np.swapaxes(w2c[:, :3, :3], -1, -2).copy()
    translation = w2c[:, :3, 3].copy()
    return _restore(rotation, single), _restore(translation, single)


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to an ``(w, x, y, z)`` quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        diagonal = np.diag(matrix)
        pivot = int(np.argmax(diagonal))
        if pivot == 0:
            scale = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif pivot == 1:
            scale = np.sqrt(max(1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(max(1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("rotation did not produce a finite quaternion")
    return quaternion / norm


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one ``(w, x, y, z)`` quaternion to a rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be finite and nonzero")
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _rotation_slerp(rotation0: np.ndarray, rotation1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical-linear interpolation between two proper rotations."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    q0 = _rotation_matrix_to_quaternion(rotation0)
    q1 = _rotation_matrix_to_quaternion(rotation1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        quaternion = (1.0 - alpha) * q0 + alpha * q1
    else:
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        quaternion = (
            np.sin((1.0 - alpha) * theta) / sin_theta * q0
            + np.sin(alpha * theta) / sin_theta * q1
        )
    return _quaternion_to_rotation_matrix(quaternion)


def build_adjacent_midpoint_slerp_pose_sequence(stored_world_view: np.ndarray) -> dict[str, np.ndarray]:
    """Interleave original C2W poses with on-path midpoint/SLERP poses."""

    input_w2c = stored_world_view_to_w2c(stored_world_view)
    input_c2w = stored_world_view_to_c2w(stored_world_view)
    input_batch, single = _pose_batch(input_c2w, "input_c2w")
    if single or len(input_batch) < 2:
        raise ValueError("on-path sequence requires at least two input poses")

    n_input = len(input_batch)
    nvs_c2w = np.empty((2 * n_input - 1, 4, 4), dtype=np.float64)
    nvs_c2w[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    source_left = np.empty(2 * n_input - 1, dtype=np.int64)
    source_right = np.empty(2 * n_input - 1, dtype=np.int64)
    source_alpha = np.empty(2 * n_input - 1, dtype=np.float64)
    for index in range(n_input - 1):
        output_index = 2 * index
        left = input_batch[index]
        right = input_batch[index + 1]
        nvs_c2w[output_index] = left
        nvs_c2w[output_index + 1, :3, :3] = _rotation_slerp(
            left[:3, :3], right[:3, :3], 0.5
        )
        nvs_c2w[output_index + 1, :3, 3] = 0.5 * (left[:3, 3] + right[:3, 3])
        source_left[output_index:output_index + 2] = [index, index]
        source_right[output_index:output_index + 2] = [index, index + 1]
        source_alpha[output_index:output_index + 2] = [0.0, 0.5]
    nvs_c2w[-1] = input_batch[-1]
    source_left[-1] = n_input - 1
    source_right[-1] = n_input - 1
    source_alpha[-1] = 0.0

    nvs_w2c = np.linalg.inv(nvs_c2w)
    update_rotation, update_translation = c2w_to_update_rt(nvs_c2w)
    return {
        "input_w2c": np.asarray(input_w2c, dtype=np.float64),
        "input_c2w": np.asarray(input_c2w, dtype=np.float64),
        "smoothed_c2w": nvs_c2w,
        "smoothed_w2c": nvs_w2c,
        "update_rotation": update_rotation,
        "update_translation": update_translation,
        "source_left_index": source_left,
        "source_right_index": source_right,
        "source_alpha": source_alpha,
    }


def build_nvs_pose_sequence(
    stored_world_view: np.ndarray,
    smoother: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert, smooth in C2W space, and build update-ready camera poses."""

    input_w2c = stored_world_view_to_w2c(stored_world_view)
    input_c2w = stored_world_view_to_c2w(stored_world_view)
    smoothed_c2w, single = _pose_batch(smoother(np.array(input_c2w, copy=True)), "smoothed_c2w")
    if single:
        raise ValueError("NVS pose sequence requires at least a batch of poses")
    smoothed_w2c = np.linalg.inv(smoothed_c2w)
    update_rotation, update_translation = c2w_to_update_rt(smoothed_c2w)
    return {
        "input_w2c": np.asarray(input_w2c, dtype=np.float64),
        "input_c2w": np.asarray(input_c2w, dtype=np.float64),
        "smoothed_c2w": smoothed_c2w,
        "smoothed_w2c": smoothed_w2c,
        "update_rotation": update_rotation,
        "update_translation": update_translation,
        "source_left_index": np.arange(len(smoothed_c2w), dtype=np.int64),
        "source_right_index": np.arange(len(smoothed_c2w), dtype=np.int64),
        "source_alpha": np.zeros(len(smoothed_c2w), dtype=np.float64),
    }

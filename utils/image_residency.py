"""Image residency policies and bounded, versioned runtime evidence.

The camera geometry device and the ground-truth image residency device are
deliberately separate.  The external fixed-pose RGB-only route uses
``cpu-stream-v1`` by default; optional pose/depth paths retain the historical
GPU residency unless a caller explicitly selects another supported policy.

This module has no dependency on the custom rasterizers.  It is therefore
usable by CPU/fake-device tests and can report CUDA allocator data when the
runtime makes it available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


IMAGE_RESIDENCY_SCHEMA = "image-residency-telemetry-v1"
IMAGE_RESIDENCY_AUTO = "auto"
IMAGE_RESIDENCY_CPU_STREAM_V1 = "cpu-stream-v1"
IMAGE_RESIDENCY_GPU_ALL_V0 = "gpu-all-v0"
SUPPORTED_IMAGE_RESIDENCIES = frozenset(
    {
        IMAGE_RESIDENCY_AUTO,
        IMAGE_RESIDENCY_CPU_STREAM_V1,
        IMAGE_RESIDENCY_GPU_ALL_V0,
    }
)


class ImageResidencyError(ValueError):
    """The requested image policy or its runtime evidence is invalid."""


def resolve_image_residency(
    requested: str | None,
    *,
    external_colmap_pose: bool,
    depth_source: str,
) -> str:
    """Resolve a versioned policy without changing optional algorithm routes."""

    policy = IMAGE_RESIDENCY_AUTO if requested in (None, "") else str(requested)
    if policy not in SUPPORTED_IMAGE_RESIDENCIES:
        raise ImageResidencyError(
            f"unsupported image_residency {policy!r}; "
            f"expected one of {sorted(SUPPORTED_IMAGE_RESIDENCIES)}"
        )
    if policy == IMAGE_RESIDENCY_AUTO:
        if external_colmap_pose and depth_source == "disabled":
            return IMAGE_RESIDENCY_CPU_STREAM_V1
        return IMAGE_RESIDENCY_GPU_ALL_V0
    if policy == IMAGE_RESIDENCY_CPU_STREAM_V1 and not (
        external_colmap_pose and depth_source == "disabled"
    ):
        raise ImageResidencyError(
            "cpu-stream-v1 is scoped to external_colmap_pose with "
            "depth_source=disabled"
        )
    return policy


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _storage_key(tensor: torch.Tensor) -> tuple[str, str, int, int]:
    """Return a key that counts safe aliases only once."""

    try:
        storage = tensor.untyped_storage()
        return (
            "storage",
            tensor.device.type,
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
    except (AttributeError, RuntimeError):
        return (
            "tensor",
            tensor.device.type,
            int(tensor.data_ptr()),
            _tensor_nbytes(tensor),
        )


def _unique_tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    total = 0
    for tensor in tensors:
        key = _storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += _tensor_nbytes(tensor)
    return total


def _cuda_memory_snapshot() -> dict[str, Any]:
    """Read allocator counters when available; never make them mandatory."""

    try:
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except (AttributeError, RuntimeError):
        return {"available": False, "status": "unavailable"}


class ImageResidencyTelemetry:
    """Static image binding plus observed transfer and allocator evidence."""

    def __init__(self, *, strategy: str, phase: str = "scene") -> None:
        if strategy not in SUPPORTED_IMAGE_RESIDENCIES - {IMAGE_RESIDENCY_AUTO}:
            raise ImageResidencyError(f"telemetry requires a resolved strategy, got {strategy!r}")
        self.strategy = strategy
        self.phase = str(phase)
        self.cameras: list[object] = []
        self.camera_order: list[str] = []
        self.camera_order_sha256: str | None = None
        self.camera_count = 0
        self.dimensions: dict[str, Any] = {}
        self.image_dtype: str | None = None
        self.cpu_resident_image_bytes = 0
        self.gpu_resident_gt_frame_count = 0
        self.gpu_resident_gt_frame_bytes = 0
        self.gpu_resident_gt_frame_count_peak = 0
        self.gpu_resident_gt_frame_bytes_peak = 0
        self.safe_alias_camera_count = 0
        self.transfer_count = 0
        self.transfer_bytes = 0
        self.cuda_memory_peak: dict[str, Any] = {"available": False}
        self.device_errors = 0

    def bind(self, cameras: Sequence[object]) -> None:
        self.cameras = list(cameras)
        self.camera_order = [str(getattr(camera, "image_name", "")) for camera in self.cameras]
        if any(not name for name in self.camera_order):
            raise ImageResidencyError("every bound camera must have an image_name")
        if len(self.camera_order) != len(set(self.camera_order)):
            raise ImageResidencyError("bound camera image_name values must be unique")
        self.camera_order_sha256 = hashlib.sha256(
            json.dumps(self.camera_order, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.camera_count = len(self.camera_order)

        shapes = {
            tuple(int(value) for value in camera.image_shape("final"))
            for camera in self.cameras
        }
        if len(shapes) == 1:
            channels, height, width = next(iter(shapes))
            self.dimensions = {
                "channels": channels,
                "height": height,
                "width": width,
            }
        else:
            self.dimensions = {
                "shape_set": [list(shape) for shape in sorted(shapes)],
            }

        tensors = [
            image
            for camera in self.cameras
            for image in camera.resident_image_tensors()
        ]
        self.cpu_resident_image_bytes = _unique_tensor_bytes(
            image for image in tensors if image.device.type == "cpu"
        )
        gpu_tensors = [image for image in tensors if image.device.type == "cuda"]
        self.gpu_resident_gt_frame_count = len(
            {
                _storage_key(image)
                for image in gpu_tensors
            }
        )
        self.gpu_resident_gt_frame_bytes = _unique_tensor_bytes(gpu_tensors)
        self.gpu_resident_gt_frame_count_peak = self.gpu_resident_gt_frame_count
        self.gpu_resident_gt_frame_bytes_peak = self.gpu_resident_gt_frame_bytes
        self.safe_alias_camera_count = sum(
            1 for camera in self.cameras if bool(getattr(camera, "images_share_storage", False))
        )
        dtypes = {str(image.dtype) for image in tensors}
        self.image_dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
        self.cuda_memory_peak = _cuda_memory_snapshot()

    def record_transfer(
        self,
        *,
        camera: object,
        image: torch.Tensor,
        stage: str,
        target: torch.device,
    ) -> None:
        del camera, stage
        if image.device.type != target.type or (
            target.index is not None and image.device.index != target.index
        ):
            self.record_device_error()
            raise ImageResidencyError(
                "image transfer returned a tensor on a different device: "
                f"target={target}, actual={image.device}"
            )
        self.transfer_count += 1
        self.transfer_bytes += _tensor_nbytes(image)
        if target.type == "cuda":
            # cpu-stream-v1 deliberately has no cache: at most this one
            # current GT tensor is owned by the caller between transfers.
            self.gpu_resident_gt_frame_count_peak = max(
                self.gpu_resident_gt_frame_count_peak, 1
            )
            self.gpu_resident_gt_frame_bytes_peak = max(
                self.gpu_resident_gt_frame_bytes_peak, _tensor_nbytes(image)
            )
        self.cuda_memory_peak = _cuda_memory_snapshot()

    def record_device_error(self) -> None:
        self.device_errors += 1

    def snapshot(self, *, phase: str | None = None) -> dict[str, Any]:
        if phase is not None:
            self.phase = str(phase)
        self.cuda_memory_peak = _cuda_memory_snapshot()
        result: dict[str, Any] = {
            "schema_version": IMAGE_RESIDENCY_SCHEMA,
            "phase": self.phase,
            "strategy": self.strategy,
            "camera_count": self.camera_count,
            "camera_order": list(self.camera_order),
            "camera_order_sha256": self.camera_order_sha256,
            "dimensions": dict(self.dimensions),
            "image_dtype": self.image_dtype,
            "cpu_resident_image_bytes": int(self.cpu_resident_image_bytes),
            "gpu_resident_gt_frame_count": int(self.gpu_resident_gt_frame_count),
            "gpu_resident_gt_frame_bytes": int(self.gpu_resident_gt_frame_bytes),
            "gpu_resident_gt_frame_count_peak": int(self.gpu_resident_gt_frame_count_peak),
            "gpu_resident_gt_frame_bytes_peak": int(self.gpu_resident_gt_frame_bytes_peak),
            "safe_alias_camera_count": int(self.safe_alias_camera_count),
            "transfer_count": int(self.transfer_count),
            "transfer_bytes": int(self.transfer_bytes),
            "device_errors": int(self.device_errors),
            "theory_is_advisory": True,
            "cuda_memory": dict(self.cuda_memory_peak),
        }
        return result

    def write(self, path: str | Path, *, phase: str | None = None) -> dict[str, Any]:
        destination = Path(path)
        if destination.exists() or destination.is_symlink():
            raise ImageResidencyError(f"residency telemetry must be absent: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot(phase=phase)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return payload

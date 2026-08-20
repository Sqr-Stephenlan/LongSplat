"""Append-only camera sampling evidence for the external fixed-pose route.

This module intentionally has no torch, CUDA, or random imports.  The trainer
owns camera selection; this helper observes the selected camera after the
existing ``randint``/``pop`` operation and records an immutable identity-bound
event.  It must never alter RNG state or the selection stack.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TELEMETRY_SCHEMA = "camera-sampling-telemetry-v1"
EVENTS_FILENAME = "camera_sampling_telemetry-v1.jsonl"
SUMMARY_FILENAME = "camera_sampling_telemetry-v1.json"


class CameraSamplingTelemetryError(ValueError):
    """The immutable camera identity or append-only telemetry contract failed."""


def _as_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CameraSamplingTelemetryError(f"{field} must be a non-empty string")
    return value


def _camera_identity(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, str], str | None]:
    internal_names_value = contract.get("image_names")
    if not isinstance(internal_names_value, list) or not internal_names_value:
        raise CameraSamplingTelemetryError("external pose contract image_names are required")
    internal_names = [_as_nonempty_string(value, "contract image name") for value in internal_names_value]
    if len(internal_names) != len(set(internal_names)):
        raise CameraSamplingTelemetryError("external pose contract image_names must be unique")

    identity = contract.get("camera_identity")
    if not isinstance(identity, Mapping):
        raise CameraSamplingTelemetryError("external pose contract camera_identity mapping is required")
    mapping_value = identity.get("basename_to_internal_name")
    if not isinstance(mapping_value, Mapping) or not mapping_value:
        raise CameraSamplingTelemetryError("camera_identity.basename_to_internal_name is required")
    basename_to_internal = {
        _as_nonempty_string(key, "camera basename"): _as_nonempty_string(value, "internal camera name")
        for key, value in mapping_value.items()
    }
    if len(basename_to_internal) != len(set(basename_to_internal.values())):
        raise CameraSamplingTelemetryError("camera identity has duplicate internal names")
    if list(basename_to_internal.values()) != internal_names:
        raise CameraSamplingTelemetryError(
            "camera identity order must match external pose contract image_names"
        )
    contract_sha = identity.get("contract_file_sha256")
    if contract_sha is not None:
        contract_sha = _as_nonempty_string(contract_sha, "camera contract SHA")
    return internal_names, basename_to_internal, contract_sha


class CameraSamplingTelemetry:
    """Record one event per already-selected external training camera."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        contract: Mapping[str, Any],
        cameras: Sequence[object],
        iterations: int,
    ) -> None:
        if int(iterations) <= 0:
            raise CameraSamplingTelemetryError("telemetry iterations must be positive")
        self.model_path = Path(model_path)
        if not self.model_path.is_dir() or self.model_path.is_symlink():
            raise CameraSamplingTelemetryError(f"telemetry model path is not a real directory: {self.model_path}")
        self.iterations = int(iterations)
        self.internal_names, self.basename_to_internal, self.contract_file_sha256 = _camera_identity(contract)
        self.internal_to_basename = {value: key for key, value in self.basename_to_internal.items()}
        active_names = [_as_nonempty_string(getattr(camera, "image_name", None), "active camera image_name") for camera in cameras]
        if active_names != self.internal_names:
            raise CameraSamplingTelemetryError(
                "active camera order must match the immutable external pose contract"
            )
        self.active_camera_count = len(active_names)
        self.internal_to_index = {name: index for index, name in enumerate(self.internal_names)}
        self.counts = {name: 0 for name in self.internal_names}
        self.event_count = 0
        self._events_path = self.model_path / EVENTS_FILENAME
        self._summary_path = self.model_path / SUMMARY_FILENAME
        if self._events_path.exists() or self._events_path.is_symlink():
            raise CameraSamplingTelemetryError(f"telemetry events file must be absent: {self._events_path}")
        if self._summary_path.exists() or self._summary_path.is_symlink():
            raise CameraSamplingTelemetryError(f"telemetry summary file must be absent: {self._summary_path}")
        self._events_file = self._events_path.open("x", encoding="utf-8")

    def record(self, iteration: int, camera: object) -> dict[str, Any]:
        """Observe one selected camera without touching RNG or camera ordering."""

        expected_iteration = self.event_count + 1
        if int(iteration) != expected_iteration:
            raise CameraSamplingTelemetryError(
                f"telemetry iteration must be sequential: expected {expected_iteration}, got {iteration}"
            )
        internal_name = _as_nonempty_string(getattr(camera, "image_name", None), "selected camera image_name")
        if internal_name not in self.internal_to_index:
            raise CameraSamplingTelemetryError(f"selected camera is absent from external contract: {internal_name}")
        self.counts[internal_name] += 1
        self.event_count += 1
        basename = self.internal_to_basename[internal_name]
        event = {
            "schema_version": TELEMETRY_SCHEMA,
            "iteration": int(iteration),
            "camera_basename": basename,
            "camera_internal_name": internal_name,
            "camera_contract_index": self.internal_to_index[internal_name],
            "exposure_count": self.counts[internal_name],
            "cumulative_unique_count": sum(1 for value in self.counts.values() if value > 0),
        }
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        self._events_file.write(line)
        self._events_file.flush()
        return event

    def finalize(self, *, checkpoint_iteration: int) -> dict[str, Any]:
        if self.event_count != self.iterations:
            raise CameraSamplingTelemetryError(
                f"telemetry event count {self.event_count} differs from iterations {self.iterations}"
            )
        self._events_file.close()
        events_sha = hashlib.sha256(self._events_path.read_bytes()).hexdigest()
        exposure_records = [
            {
                "camera_basename": self.internal_to_basename[name],
                "camera_internal_name": name,
                "camera_contract_index": index,
                "exposure_count": self.counts[name],
            }
            for index, name in enumerate(self.internal_names)
        ]
        exposures = [record["exposure_count"] for record in exposure_records]
        summary = {
            "schema_version": TELEMETRY_SCHEMA,
            "selection_policy": "one_camera_per_iteration_random_pop_without_replacement_per_full_stack",
            "rng_observation": "telemetry makes no random calls and does not mutate RNG state",
            "active_camera_count": self.active_camera_count,
            "active_camera_order": [self.internal_to_basename[name] for name in self.internal_names],
            "active_camera_internal_order": list(self.internal_names),
            "camera_contract_file_sha256": self.contract_file_sha256,
            "iterations": self.iterations,
            "sampled_iteration_count": self.event_count,
            "unique_camera_count": sum(1 for value in exposures if value > 0),
            "coverage_fraction": sum(1 for value in exposures if value > 0) / self.active_camera_count,
            "complete_rounds": self.iterations // self.active_camera_count,
            "partial_round_size": self.iterations % self.active_camera_count,
            "min_exposure_count": min(exposures),
            "max_exposure_count": max(exposures),
            "exposure_counts": exposure_records,
            "events_path": str(self._events_path),
            "events_sha256": events_sha,
            "events_size_bytes": self._events_path.stat().st_size,
            "checkpoint_iteration": int(checkpoint_iteration),
        }
        with self._summary_path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        summary["summary_path"] = str(self._summary_path)
        summary["summary_sha256"] = hashlib.sha256(self._summary_path.read_bytes()).hexdigest()
        summary["summary_size_bytes"] = self._summary_path.stat().st_size
        return summary

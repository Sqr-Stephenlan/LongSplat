from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scene.cameras import Camera, MiniCam
from utils.image_residency import (
    IMAGE_RESIDENCY_CPU_STREAM_V1,
    IMAGE_RESIDENCY_GPU_ALL_V0,
    ImageResidencyError,
    ImageResidencyTelemetry,
    resolve_image_residency,
)


def _camera(
    name: str,
    *,
    width: int,
    height: int,
    alpha: bool = False,
    alpha_value: float = 1.0,
    residency: str = IMAGE_RESIDENCY_CPU_STREAM_V1,
    telemetry: ImageResidencyTelemetry | None = None,
) -> Camera:
    image = torch.linspace(
        0.0,
        1.0,
        3 * height * width,
        dtype=torch.float32,
    ).reshape(3, height, width)
    mask = torch.full((1, height, width), alpha_value, dtype=torch.float32) if alpha else None
    return Camera(
        colmap_id=0,
        R=None,
        T=None,
        FoVx=None,
        FoVy=None,
        image=image,
        gt_alpha_mask=mask,
        image_name=name,
        uid=0,
        data_device="cpu",
        disable_resize=True,
        image_residency=residency,
        residency_telemetry=telemetry,
    )


class ImageResidencyTests(unittest.TestCase):
    def setUp(self) -> None:
        # These are CPU-only tests.  Telemetry's optional allocator probe is
        # patched so importing/running this suite never queries CUDA.
        self.cuda_probe = mock.patch(
            "utils.image_residency._cuda_memory_snapshot",
            return_value={"available": False},
        )
        self.cuda_probe.start()
        self.addCleanup(self.cuda_probe.stop)

    def test_cpu_stream_binds_dynamic_camera_order_shapes_bytes_and_dtype(self) -> None:
        for count, width, height in ((1, 8, 6), (3, 11, 7), (17, 5, 13)):
            with self.subTest(count=count, width=width, height=height):
                telemetry = ImageResidencyTelemetry(strategy=IMAGE_RESIDENCY_CPU_STREAM_V1)
                cameras = [
                    _camera(f"camera-{index}", width=width, height=height, telemetry=telemetry)
                    for index in range(count)
                ]
                telemetry.bind(cameras)

                evidence = telemetry.snapshot(phase="training")

                self.assertEqual(evidence["camera_count"], count)
                self.assertEqual(evidence["camera_order"], [f"camera-{index}" for index in range(count)])
                self.assertEqual(evidence["dimensions"], {"channels": 3, "height": height, "width": width})
                self.assertEqual(evidence["image_dtype"], "torch.float32")
                self.assertEqual(evidence["cpu_resident_image_bytes"], count * 3 * height * width * 4)
                self.assertEqual(evidence["gpu_resident_gt_frame_count"], 0)
                self.assertEqual(evidence["gpu_resident_gt_frame_bytes"], 0)
                self.assertEqual(evidence["safe_alias_camera_count"], count)
                self.assertTrue(evidence["theory_is_advisory"])

    def test_telemetry_reports_shape_set_for_arbitrary_camera_shapes(self) -> None:
        telemetry = ImageResidencyTelemetry(strategy=IMAGE_RESIDENCY_CPU_STREAM_V1)
        cameras = [
            _camera("small", width=8, height=6, telemetry=telemetry),
            _camera("wide", width=11, height=7, telemetry=telemetry),
        ]
        telemetry.bind(cameras)

        self.assertEqual(
            telemetry.snapshot()["dimensions"],
            {"shape_set": [[3, 6, 8], [3, 7, 11]]},
        )

    def test_safe_alias_is_disabled_when_alpha_changes_optimization_semantics(self) -> None:
        aliased = _camera("plain", width=8, height=6, alpha=False)
        masked = _camera("masked", width=8, height=6, alpha=True, alpha_value=0.5)

        self.assertTrue(aliased.images_share_storage)
        self.assertEqual(aliased.original_image.data_ptr(), aliased.original_image_final.data_ptr())
        self.assertFalse(masked.images_share_storage)
        self.assertNotEqual(masked.original_image.data_ptr(), masked.original_image_final.data_ptr())
        self.assertFalse(torch.equal(masked.original_image, masked.original_image_final))
        self.assertEqual(masked.original_image.dtype, masked.original_image_final.dtype)

    def test_to_final_switches_to_final_pixels_without_device_transfer(self) -> None:
        camera = _camera("masked", width=7, height=5, alpha=True, alpha_value=0.25)
        final = camera.original_image_final.clone()

        camera.to_final()

        self.assertTrue(torch.equal(camera.original_image, final))
        self.assertEqual(camera.image_shape("final"), (3, 5, 7))
        self.assertEqual(camera.original_image.device.type, "cpu")
        self.assertEqual(camera.original_image.dtype, torch.float32)

    def test_cpu_stream_transfer_does_not_mutate_camera_residency(self) -> None:
        telemetry = ImageResidencyTelemetry(strategy=IMAGE_RESIDENCY_CPU_STREAM_V1)
        camera = _camera("one", width=8, height=6, telemetry=telemetry)
        telemetry.bind([camera])

        transferred = camera.get_image(device="meta")

        self.assertEqual(transferred.device.type, "meta")
        self.assertEqual(camera.original_image.device.type, "cpu")
        self.assertEqual(camera.original_image_final.device.type, "cpu")
        self.assertEqual(telemetry.snapshot()["transfer_count"], 1)
        self.assertGreater(telemetry.snapshot()["transfer_bytes"], 0)
        self.assertEqual(telemetry.snapshot()["gpu_resident_gt_frame_count"], 0)

    def test_transfer_device_mismatch_is_hard_error(self) -> None:
        telemetry = ImageResidencyTelemetry(strategy=IMAGE_RESIDENCY_CPU_STREAM_V1)
        camera = _camera("one", width=4, height=3, telemetry=telemetry)
        telemetry.bind([camera])

        with self.assertRaisesRegex(ImageResidencyError, "different device"):
            telemetry.record_transfer(
                camera=camera,
                image=camera.original_image,
                stage="optimization",
                target=SimpleNamespace(type="meta", index=None),
            )
        self.assertEqual(telemetry.snapshot()["device_errors"], 1)

    def test_policy_auto_is_route_scoped_and_optional_route_stays_legacy(self) -> None:
        self.assertEqual(
            resolve_image_residency(
                "auto", external_colmap_pose=True, depth_source="disabled"
            ),
            IMAGE_RESIDENCY_CPU_STREAM_V1,
        )
        self.assertEqual(
            resolve_image_residency(
                "auto", external_colmap_pose=False, depth_source="mast3r"
            ),
            IMAGE_RESIDENCY_GPU_ALL_V0,
        )
        with self.assertRaisesRegex(ImageResidencyError, "scoped"):
            resolve_image_residency(
                IMAGE_RESIDENCY_CPU_STREAM_V1,
                external_colmap_pose=False,
                depth_source="mast3r",
            )

    def test_minicam_has_renderer_geometry_but_no_ground_truth_residency(self) -> None:
        world_view = torch.eye(4)
        camera = MiniCam(
            width=11,
            height=7,
            fovy=1.0,
            fovx=1.2,
            znear=0.01,
            zfar=100.0,
            world_view_transform=world_view,
            full_proj_transform=world_view,
        )

        self.assertEqual((camera.image_width, camera.image_height), (11, 7))
        self.assertEqual(camera.camera_center.shape, (3,))
        self.assertTrue(hasattr(camera, "projection_matrix"))
        self.assertTrue(hasattr(camera, "cam_rot_delta"))
        self.assertFalse(hasattr(camera, "original_image"))
        self.assertFalse(hasattr(camera, "get_image"))

    def test_nvs_uses_bounded_gt_writer_and_minicam(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "render.py").read_text(encoding="utf-8")
        nvs_body = source.split("def render_nvs", 1)[1].split("def render_set", 1)[0]

        self.assertNotIn("gt_list", nvs_body)
        self.assertIn("get_writer", nvs_body)
        self.assertIn("MiniCam", nvs_body)
        self.assertIn('get_image(stage="final", device="cpu")', nvs_body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

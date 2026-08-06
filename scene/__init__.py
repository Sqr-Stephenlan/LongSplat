# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
import torch
from utils.mast3r_utils import Mast3rMatcher
from scene.gaussian_model import BasicPointCloud
from torch.nn import functional as F
from utils.external_colmap_pose import external_camera_contract

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.load_pose:
            print("Loading Cameras with Pose")
            scene_info = sceneLoadTypeCallbacks["Eval"](args.source_path, args.model_path, args.images, args.eval)
        elif args.mode == "custom":
            print("Loading Cameras with Custom Pose")
            scene_info = sceneLoadTypeCallbacks["Custom"](args.source_path, args.images, args.eval)
        elif args.mode == "free":
            print("Loading Cameras with Free Pose")
            scene_info = sceneLoadTypeCallbacks["Free"](args.source_path, args.images, args.eval)
        elif args.mode == "tanks":
            print("Loading Cameras with Tanks Pose")
            scene_info = sceneLoadTypeCallbacks["Tanks"](args.source_path, args.images, args.eval)
        elif args.mode == "hike":
            print("Loading Cameras with Hike Pose")
            scene_info = sceneLoadTypeCallbacks["Hike"](args.source_path, args.images, args.eval)
        elif args.mode == "co3d":
            print("Loading Cameras with Co3d Pose")
            scene_info = sceneLoadTypeCallbacks["Co3d"](args.source_path, args.images, args.eval)
        else:
            print("Loading Cameras without Pose")
            scene_info = sceneLoadTypeCallbacks["Unposed"](args.source_path, args.images, args.eval)

        self.gaussians.set_appearance(len(scene_info.train_cameras))

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            for cam in self.getAllCameras():
                if cam.R_pred is not None and cam.T_pred is not None:
                    cam.update_RT(cam.R_pred, cam.T_pred)
                cam.to_final()
        else:
            if getattr(args, "external_colmap_pose", False):
                self._initialize_external_colmap_scene(scene_info, args)
                return

            self.init_frame_num = args.init_frame_num
            matcher = Mast3rMatcher()
            source_path = os.path.join(args.source_path, args.images)
            camera_paths = [os.path.join(source_path, image_name) for image_name in os.listdir(source_path)]
            camera_paths.sort()
            
            init_camera_paths = []
            for cam in self.getTrainCameras():
                for camera_path in camera_paths:
                    if cam.image_name in camera_path:
                        init_camera_paths.append(camera_path)
                        break
                if len(init_camera_paths) == self.init_frame_num:
                    break

            if self.getTrainCameras()[0].intrinsic is not None:
                intrinsic_np = self.getTrainCameras()[0].intrinsic.detach().cpu().numpy()
            else:
                print("Intrinsic not found, using Mast3r focal")
                intrinsic_np = None
            
            pts3d, world2cam, depth_maps, focal_length = matcher.global_align(init_camera_paths, intrinsic_np)

            if intrinsic_np is None:
                for cam in self.getAllCameras():
                    cam.update_focal(focal_length)

            for i, Rt in enumerate(world2cam):
                R = Rt[:3, :3].t()
                T = Rt[:3, 3]
                cur_viewpoint_cam = self.getTrainCameras()[i]
                cur_viewpoint_cam.update_RT(R, T)
                if i == 0:
                    next_viewpoint_cam = self.getTrainCameras()[i + 1]
                    intrinsic_np = cur_viewpoint_cam.intrinsic.detach().cpu().numpy()
                    cur_viewpoint_cam.kp0, cur_viewpoint_cam.kp1, desc_conf1, desc_conf2, pts3d1, pts3d2, conf1, conf2, depth1, depth2 = matcher._forward(
                        cur_viewpoint_cam.original_image,
                        next_viewpoint_cam.original_image,
                        intrinsic_np
                    )
                    cur_viewpoint_cam.depth_map = torch.from_numpy(depth_maps[i]).float().cuda().detach()
                    cur_viewpoint_cam.depth_map = F.interpolate(cur_viewpoint_cam.depth_map.unsqueeze(0).unsqueeze(0), 
                                                                size=(cur_viewpoint_cam.image_height, 
                                                                      cur_viewpoint_cam.image_width), 
                                                                mode='bilinear', 
                                                                align_corners=False).squeeze(0).squeeze(0)

                else:
                    pre_viewpoint_cam = self.getTrainCameras()[i - 1]
                    
                    intrinsic_np = cur_viewpoint_cam.intrinsic.detach().cpu().numpy()
                    cur_viewpoint_cam.kp0, cur_viewpoint_cam.kp1, desc_conf1, desc_conf2, pts3d1, pts3d2, conf1, conf2, depth1, depth2 = matcher._forward(
                        pre_viewpoint_cam.original_image, 
                        cur_viewpoint_cam.original_image, 
                        intrinsic_np
                    )
                    cur_viewpoint_cam.conf = torch.ones(cur_viewpoint_cam.kp0.shape[0], device=cur_viewpoint_cam.kp0.device)
                    cur_viewpoint_cam.depth_map = torch.from_numpy(depth_maps[i]).float().cuda().detach()
                    cur_viewpoint_cam.depth_map = F.interpolate(cur_viewpoint_cam.depth_map.unsqueeze(0).unsqueeze(0), 
                                                                size=(cur_viewpoint_cam.image_height, 
                                                                      cur_viewpoint_cam.image_width), 
                                                                mode='bilinear', 
                                                                align_corners=False).squeeze(0).squeeze(0)
                cur_viewpoint_cam.is_registered = True
            
            self.cameras_extent = 10.0
            print(f'self.cameras_extent: {self.cameras_extent}')
            self.gaussians.create_from_pcd(BasicPointCloud(points=pts3d, colors=None, normals=None), self.cameras_extent)

    def _initialize_external_colmap_scene(self, scene_info, args):
        """Initialize one isolated fixed-pose COLMAP route; default is off."""

        if not getattr(args, "disable_resize", False):
            raise ValueError("external_colmap_pose requires --disable_resize for the 1280x720 contract")
        if getattr(args, "depth_source", "disabled") != "disabled":
            raise ValueError("external_colmap_pose requires --depth_source disabled")
        cameras = self.getAllCameras()
        expected_names = [f"frame_{index:06d}" for index in range(len(cameras))]
        contract = external_camera_contract(
            scene_info.train_cameras + scene_info.test_cameras,
            expected_names=expected_names,
        )
        if scene_info.point_cloud is None:
            raise ValueError("external_colmap_pose requires COLMAP sparse points")

        for camera in cameras:
            if camera.R_gt is None or camera.T_gt is None:
                raise ValueError(f"missing COLMAP pose for {camera.image_name}")
            camera.update_RT(
                camera.R_gt.to(device=camera.data_device, dtype=torch.float32),
                camera.T_gt.to(device=camera.data_device, dtype=torch.float32),
            )
            camera.cam_rot_delta.requires_grad_(False)
            camera.cam_trans_delta.requires_grad_(False)
            camera.depth_map = None
            camera.pre_depth_map = None
            camera.kp0 = None
            camera.kp1 = None
            camera.conf = None
            camera.is_registered = True

        self.init_frame_num = len(cameras)
        self.external_colmap_contract = {
            **contract,
            "external_colmap_pose": True,
            "mast3r_global_align_skipped": True,
            "pose_frozen": True,
            "vda_external_depth_disabled": True,
            "active_image_width": cameras[0].image_width,
            "active_image_height": cameras[0].image_height,
            "active_focal_x_px": float(cameras[0].Focalx),
            "active_focal_y_px": float(cameras[0].Focaly),
            "depth_source": "disabled",
        }
        print(
            "EXTERNAL_POSE_CONTRACT "
            + json.dumps(self.external_colmap_contract, sort_keys=True, allow_nan=False),
            flush=True,
        )
        self.cameras_extent = 10.0
        print(f"self.cameras_extent: {self.cameras_extent}")
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getAllCameras(self, scale=1.0):
        return self.getTrainCameras(scale) + self.getTestCameras(scale)
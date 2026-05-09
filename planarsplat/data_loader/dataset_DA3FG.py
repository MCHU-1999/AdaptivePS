import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import open3d as o3d
from PIL import Image

from utils.model_util import get_K_Rt_from_P
from utils.mesh_util import refuse_mesh
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.mesh_util import render_depths
import math
from loguru import logger
from typing import NamedTuple, List, Dict
import trimesh

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int

class ViewInfo(nn.Module):
    def __init__(self, cam_info: Dict, gt_info: Dict):
        super().__init__()
        # get cam info
        self.intrinsic = cam_info['intrinsic'].cuda()
        self.pose = cam_info['pose'].cuda()
        self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
        self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
        self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
        self.raster_cam_center = cam_info['raster_cam_center'].cuda()
        self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
        self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
        self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
        self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
        self.raster_img_center = cam_info['raster_img_center'].cuda()
        self.cam_loc = cam_info['cam_loc'].cuda()

        # Store paths for lazy loading
        self.image_path = gt_info['image_path']
        self.depth_path = gt_info['depth_path']
        self.normal_path = gt_info['normal_path']
        self.fg_mask_path = gt_info['fg_mask_path']
        self.index = gt_info['index']
        
        # Store image resolution and scene bounding sphere for processing
        self.img_res = gt_info['img_res']
        
        # Cache for loaded data (will be loaded on demand)
        self._rgb_cache = None
        self._mono_depth_cache = None
        self._mono_normal_global_cache = None
        self._fg_mask_cache = None

        # other info
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None
    
    def _load_rgb(self):
        """Lazy load RGB image"""
        if self._rgb_cache is None:
            rgb = np.array(Image.open(self.image_path))
            rgb = torch.from_numpy(rgb).cuda().float() / 255.0  # h, w, 3
            self._rgb_cache = rgb.reshape(-1, 3)  # hw, 3
        return self._rgb_cache
    
    def _load_depth(self):
        """Lazy load depth map"""
        if self._mono_depth_cache is None:
            depth = np.load(self.depth_path)
            depth = torch.from_numpy(depth).cuda().float()  # h, w
            self._mono_depth_cache = depth.reshape(-1)  # hw
        return self._mono_depth_cache
    
    def _load_normals(self):
        """In a DA3 dataset thed normals are actually in world coordinates and raw"""
        if self._mono_normal_global_cache is None:
            normal = np.load(self.normal_path)  # Shape: (H, W, 3)
            normal_global = torch.from_numpy(normal).cuda().float()
            normal_global = normal_global.view(-1, 3)
            
            self._mono_normal_global_cache = normal_global
        return self._mono_normal_global_cache
    
    def _load_fg_mask(self):
        """Lazy load fg_mask"""
        if self._fg_mask_cache is None:
            fg_mask = np.array(Image.open(self.fg_mask_path))  # h, w, 1
            fg_mask = (fg_mask > 0.5).astype(bool)
            fg_mask = torch.from_numpy(fg_mask).cuda().bool()
            self._fg_mask_cache = fg_mask.reshape(-1)  # hw
        return self._fg_mask_cache
    
    @property
    def rgb(self):
        return self._load_rgb()
    
    @property
    def mono_depth(self):
        return self._load_depth()
    
    @property
    def mono_normal_local(self):
        # This property is not actively used by the trainer but can be derived if needed.
        normal_global = self._load_normals()
        return normal_global @ self.pose[:3, :3]  # Transform back to local if needed
    
    @property
    def mono_normal_global(self):
        return self._load_normals()
    
    @property
    def fg_mask(self):
        return self._load_fg_mask()

class SceneDatasetDemo:
    def __init__(
        self,
        data,
        img_res: List,
        dataset_name: str = 'DA3FG',
        tag: str = 'example',
        voxel_length: float=0.05,
        sdf_trunc: float=0.08,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.tag = tag
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res  # [height, width]

        # Expect paths to be provided for lazy loading
        image_paths = data['image_paths']
        depth_paths = data['depth_paths']
        normal_paths = data['normal_paths']
        fg_mask_paths = data.get('fg_mask_paths', None)
        expdir = data['expdir']
        self.mono_mesh_dest = os.path.join(expdir, 'mono_mesh.ply')
        
        self.n_images = len(image_paths)
        
        # Validate that the number of paths matches the number of images
        assert len(depth_paths) == self.n_images, f"Depth paths count mismatch: {len(depth_paths)} vs {self.n_images}"
        assert len(normal_paths) == self.n_images, f"Normal paths count mismatch: {len(normal_paths)} vs {self.n_images}"

        if fg_mask_paths is not None:
            assert len(fg_mask_paths) == self.n_images, f"Mask paths count mismatch: {len(fg_mask_paths)} vs {self.n_images}"

            all_depths = []
            for depth_path, fg_mask_path in zip(depth_paths, fg_mask_paths):
                depth = np.load(depth_path)
                fg_mask = np.array(Image.open(fg_mask_path))  # h, w, 1
                fg_mask = (fg_mask > 0.5).astype(bool)
                fg_mask = np.squeeze(fg_mask)

                if np.all(fg_mask):
                    # mask is all true, don't count
                    continue
                
                valid_depth = depth[(depth > 0) & fg_mask]
                p95 = np.percentile(valid_depth, 95)
                
                all_depths.append(p95)
        else:
            raise NotImplementedError("Well sorry you gotta have masks") 
        
        # Put depth_trunc into dataset class
        if len(all_depths) > 0:
            self.depth_trunc = np.mean(all_depths)
        else:
            self.depth_trunc = kwargs.get('depth_trunc', 10.0)

        # Load camera parameters (lightweight)
        self.intrinsics_all = [torch.from_numpy(intrinsic).cuda() for intrinsic in data['intrinsics']]
        self.poses_all = [torch.from_numpy(extrinsic).cuda() for extrinsic in data['extrinsics']]
        
        # Store paths for lazy loading in ViewInfo
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.normal_paths = normal_paths

        # --- Efficient Mesh Generation ---
        # Load depth maps as CPU numpy arrays — refuse_mesh uses Open3D (CPU-only),
        # so there is no reason to push these to GPU VRAM.
        logger.info('Loading depth maps for mesh generation...')
        mono_depths = [np.load(depth_path).reshape(img_res[0], img_res[1]).astype(np.float32)
                       for depth_path in depth_paths]
        logger.info(f'Loaded {len(mono_depths)} depth maps for mesh generation')

        # Generate and save the initial coarse mesh from monocular depth maps.
        mesh = refuse_mesh(
            mono_depths,
            [x.cpu().numpy() for x in self.poses_all],
            [x.cpu().numpy() for x in self.intrinsics_all],
            img_res[0],
            img_res[1],
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            depth_trunc=self.depth_trunc
        )
        o3d.io.write_triangle_mesh(self.mono_mesh_dest, mesh)
        del mono_depths  # free CPU RAM before training starts

        self.raster_cam_w2c_list, self.raster_cam_proj_list, self.raster_cam_fullproj_list, self.raster_cam_center_list, self.raster_cam_FovX_list, self.raster_cam_FovY_list, self.raster_img_center_list = self.get_raster_cameras(
            self.intrinsics_all, self.poses_all, img_res[0], img_res[1])
        
        # Prepare the list of views, which will lazy-load data as needed.
        self.view_info_list = []
        for idx in tqdm(range(self.n_images), desc='building view list...'):
            cam_loc = self.poses_all[idx][:3, 3].clone()            
            cam_info = {
                "intrinsic": self.intrinsics_all[idx].clone(),
                "pose": self.poses_all[idx].clone(),  # camera to world
                "raster_cam_w2c": self.raster_cam_w2c_list[idx].clone(),
                "raster_cam_proj": self.raster_cam_proj_list[idx].clone(),
                "raster_cam_fullproj": self.raster_cam_fullproj_list[idx].clone(),
                "raster_cam_center": self.raster_cam_center_list[idx].clone(),
                "raster_cam_FovX": self.raster_cam_FovX_list[idx].clone(),
                "raster_cam_FovY": self.raster_cam_FovY_list[idx].clone(),
                "raster_img_center": self.raster_img_center_list[idx].clone(),
                "cam_loc": cam_loc.squeeze(0),
            }

            gt_info = {
                "image_path": image_paths[idx],
                "depth_path": self.depth_paths[idx],
                "normal_path": self.normal_paths[idx],
                "img_res": img_res,
                'index': idx,
                'fg_mask_path': fg_mask_paths[idx]
            }
            self.view_info_list.append(ViewInfo(cam_info, gt_info))

        logger.info('data loader finished')
    
    def load_cameras(self, cam_dict, n_images, debug_start_idx=-1):
        if debug_start_idx == -1:
            scale_mats = [cam_dict['scale_mat_%d' % idx].to(dtype=torch.float32) for idx in range(n_images)]
            world_mats = [cam_dict['world_mat_%d' % idx].to(dtype=torch.float32) for idx in range(n_images)]
        else:
            scale_mats = [cam_dict['scale_mat_%d' % (debug_start_idx + idx)].to(dtype=torch.float32) for idx in range(n_images)]
            world_mats = [cam_dict['world_mat_%d' % (debug_start_idx + idx)].to(dtype=torch.float32) for idx in range(n_images)]

        intrinsics_all = []
        poses_all = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, pose = get_K_Rt_from_P(None, P.numpy())
            intrinsics_all.append(torch.from_numpy(intrinsic).float().cuda())
            poses_all.append(torch.from_numpy(pose).float().cuda())
        
        return intrinsics_all, poses_all
    
    def get_raster_cameras(self, intrinsics_all, poses_all, height, width):
        # zfar = 100.
        zfar = 250.
        znear = 0.01
        raster_cam_w2c_list = []
        raster_cam_proj_list = []
        raster_cam_fullproj_list = []
        raster_cam_center_list = []
        raster_cam_FovX_list = []
        raster_cam_FovY_list = []
        raster_img_center_list = []

        for i in range(self.n_images):
            focal_length_x = intrinsics_all[i][0,0]
            focal_length_y = intrinsics_all[i][1,1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            cx = intrinsics_all[i][0, 2]
            cy = intrinsics_all[i][1, 2]

            c2w = poses_all[i]  # 4, 4
            w2c = c2w.inverse()  # 4, 4
            w2c_right = w2c.T

            world_view_transform = w2c_right.clone()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            raster_cam_w2c_list.append(world_view_transform)
            raster_cam_proj_list.append(projection_matrix)
            raster_cam_fullproj_list.append(full_proj_transform)
            raster_cam_center_list.append(camera_center)
            raster_cam_FovX_list.append(torch.tensor([FovX]).cuda())
            raster_cam_FovY_list.append(torch.tensor([FovY]).cuda())

            raster_img_center_list.append(torch.tensor([cx, cy]).cuda())
        
        return raster_cam_w2c_list, raster_cam_proj_list, raster_cam_fullproj_list, raster_cam_center_list, raster_cam_FovX_list, raster_cam_FovY_list, raster_img_center_list

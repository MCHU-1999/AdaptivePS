import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import open3d as o3d

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
        self.index = gt_info['index']
        
        # Store image resolution and scene bounding sphere for processing
        self.img_res = gt_info['img_res']
        
        # Cache for loaded data (will be loaded on demand)
        self._rgb_cache = None
        self._mono_depth_cache = None
        self._mono_normal_global_cache = None

        # other info
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None
    
    def _load_rgb(self):
        """Lazy load RGB image"""
        if self._rgb_cache is None:
            from PIL import Image
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
        """In a Blender dataset thed normals are actually in world coordinates and raw"""
        if self._mono_normal_global_cache is None:
            normal = np.load(self.normal_path)  # Shape: (H, W, 3)
            normal_global = torch.from_numpy(normal).cuda().float()
            normal_global = normal_global.view(-1, 3)
            
            self._mono_normal_global_cache = normal_global
        return self._mono_normal_global_cache
    
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

class SceneDatasetDemo:
    def __init__(
        self,
        data,
        img_res: List,
        dataset_name: str = 'blender',
        tag: str = 'example',
        mesh_pre_align: bool = False,
        voxel_length: float=0.05,
        sdf_trunc: float=0.08,
        depth_trunc: float = 5.0,
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
        expdir = data['expdir']
        self.mono_mesh_dest = os.path.join(expdir, 'mono_mesh.ply')
        
        self.n_images = len(image_paths)
        
        # Validate that the number of paths matches the number of images
        assert len(depth_paths) == self.n_images, f"Depth paths count mismatch: {len(depth_paths)} vs {self.n_images}"
        assert len(normal_paths) == self.n_images, f"Normal paths count mismatch: {len(normal_paths)} vs {self.n_images}"

        # Load camera parameters (lightweight)
        self.intrinsics_all = [torch.from_numpy(intrinsic).cuda() for intrinsic in data['intrinsics']]
        self.poses_all = [torch.from_numpy(extrinsic).cuda() for extrinsic in data['extrinsics']]
        
        # Store paths for lazy loading in ViewInfo
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.normal_paths = normal_paths

        # --- Efficient Mesh Generation ---
        # Temporarily load all depth maps just for the initial mesh fusion.
        print("Loading depth maps for mesh generation...")
        mono_depths = []
        for depth_path in depth_paths:
            depth = np.load(depth_path)
            depth_tensor = torch.from_numpy(depth).cuda().float()  # h, w
            mono_depths.append(depth_tensor)
        
        print(f"Loaded {len(mono_depths)} depth maps for mesh generation")

        # Generate and save the initial coarse mesh from monocular depth maps.
        mesh = refuse_mesh(
            [x.cpu().squeeze().reshape(img_res[0], img_res[1]).numpy() for x in mono_depths],
            [x.cpu().numpy() for x in self.poses_all],
            [x.cpu().numpy() for x in self.intrinsics_all],
            img_res[0],
            img_res[1],
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc
        )
        o3d.io.write_triangle_mesh(self.mono_mesh_dest, mesh)

        # --- Optional Pre-alignment of Depth Maps ---
        if mesh_pre_align:
            mesh = trimesh.load_mesh(self.mono_mesh_dest)
            absolute_img_path = os.path.abspath(image_paths[0])
            current_dir = os.path.dirname(absolute_img_path)
            parent_dir = os.path.dirname(current_dir)
            aligned_depth_dir = os.path.join(parent_dir, 'aligned_depth')

            rendered_depths = render_depths(mesh, [pose.cpu().numpy() for pose in self.poses_all], [intrinsic.cpu().numpy()[:3, :3] for intrinsic in self.intrinsics_all], H=img_res[0], W=img_res[1])
            rendered_depths = [torch.from_numpy(rd).reshape(-1).float() for rd in rendered_depths]
            from utils.align import align_depth_scale
            
            # Apply alignment corrections and save the new depth maps to a separate directory.
            print("Applying depth alignment corrections...")
            os.makedirs(aligned_depth_dir, exist_ok=True)
            aligned_depth_paths = []
            
            for i in tqdm(range(len(mono_depths)), desc='aligning depth...'):
                md = mono_depths[i].cuda()
                rd = rendered_depths[i].cuda()
                weight = ((md.reshape(-1) - rd).abs() <= 0.5) & (rd > 0.05)
                d_scale = align_depth_scale(md.reshape(1, -1), rd.reshape(1, -1), weight=weight.reshape(1, -1).float())
                if d_scale > 0:
                    md = md * d_scale.item()
                    md = torch.clamp(md, 0, 300)
                
                # Save aligned depth map
                img_name = os.path.basename(depth_paths[i]).rsplit('.', 1)[0]
                aligned_depth_path = os.path.join(aligned_depth_dir, f'{img_name}.npy')
                np.save(aligned_depth_path, md.cpu().numpy().astype(np.float32))
                aligned_depth_paths.append(aligned_depth_path)
            
            # Update the dataset's depth paths to point to the newly aligned depth maps.
            self.depth_paths = aligned_depth_paths
            print(f"Saved {len(aligned_depth_paths)} aligned depth maps")
        
        # --- Memory Management ---
        # Clear the temporarily loaded depth data to free up GPU memory.
        del mono_depths
        torch.cuda.empty_cache()
        print("Cleared temporary depth data from memory")

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
                'index': idx
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
        zfar = 100.
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

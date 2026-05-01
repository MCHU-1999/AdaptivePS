import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import open3d as o3d
import numpy as np
from tqdm import tqdm
import numpy as np
from typing import List
import glob
import torch
import pyrender

def ground_detection(mesh):
    pass

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print("post processing the mesh to have {} clusters".format(cluster_to_keep))
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh.remove_triangles_by_mask(triangles_to_remove)  # ← Modify original mesh
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    
    print("num vertices post {}".format(len(mesh.vertices)))
    return mesh

def refuse_mesh(
    depths: List[np.ndarray], 
    poses: List[np.ndarray], 
    intrinsics: List[np.ndarray], 
    H: int, 
    W: int, 
    voxel_length: float=0.05, 
    sdf_trunc: float=0.08, 
    depth_trunc: float=5.0,
    depth_scale: float=1.0
):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose, K, depth_pred in tqdm(zip(poses, intrinsics, depths)):
        depth_pred = depth_pred.reshape(H, W)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()

    # return mesh
    return post_process_mesh(mesh)

def get_coarse_mesh(
    net, 
    view_info_list: List, 
    H: int, 
    W: int, 
    voxel_length: float=0.05, 
    sdf_trunc: float=0.08,
    depth_trunc: float=5.0
):
    """
    Generates a coarse mesh by rendering depth maps from the current state of the planes and then 
    fusing them using a TSDF volume. The TSDF fusion acts as a robust regularizer. It averages out
    noise, removes outliers, and creates a cleaned-up "consensus" geometry of the scene.
    This resulting mesh is then used as a geometric prior for subsequent merging steps.
    """
    scene_scale = net.planarSplat.pose_cfg.scale
    scene_offset = net.planarSplat.pose_cfg.offset
    poses = []
    intrinsics = []
    for view_info in view_info_list:
        pose = view_info.pose.clone()
        pose[:3, 3] /= scene_scale
        pose[:3, 3] += torch.tensor(scene_offset).to(pose.device)
        poses.append(pose.cpu().numpy())
        intrinsics.append(view_info.intrinsic[:3, :3].cpu().numpy())

    depths = []
    for iter in range(len(view_info_list)):
        with torch.no_grad():
            allmap = net.planarSplat(view_info_list[iter], 50000)
        # get rendered maps
        depth = allmap[0:1].squeeze().reshape(H, W).cpu().numpy() / scene_scale
        depths.append(depth)

    mesh = refuse_mesh(depths, poses, intrinsics, H, W, voxel_length=voxel_length, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    return mesh

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES
        self.render_flags = pyrender.RenderFlags.FACE_NORMALS
        
    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        # return self.renderer.render(self.scene)
        return self.renderer.render(self.scene, self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

# def render_depths(mesh, poses, intrinsics, H, W):
#     renderer = Renderer(height=H, width=W)
#     mesh_opengl = renderer.mesh_opengl(mesh)
#     rendered_depths = []
#     try:
#         for pose, K in tqdm(zip(poses, intrinsics)):
#             intrinsic = np.eye(4)
#             intrinsic[:3, :3] = K
#             _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
#             rendered_depths.append(depth_pred)
#     finally:
#         renderer.delete()

#     return rendered_depths


class FastRenderer():
    def __init__(self, mesh, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene(bg_color=[1, 1, 1, 1])
        
        # Add mesh ONCE during initialization
        self.mesh_node = self.scene.add(mesh)
        
        # Add a placeholder camera node ONCE
        # Will update its pose and intrinsics later
        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0) # Placeholder
        self.cam_node = self.scene.add(self.cam, pose=np.eye(4))
        
        self.render_flags = pyrender.RenderFlags.FLAT

    def render_frame(self, height, width, intrinsics, pose):
        # Update viewport if it changed (usually static in a batch)
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        
        # Update Camera Intrinsics 
        # (Updating the camera object is faster than adding/removing)
        new_cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2], cy=intrinsics[1, 2],
            fx=intrinsics[0, 0], fy=intrinsics[1, 1]
        )
        self.scene.main_camera_node.camera = new_cam
        
        # Update Camera Pose
        self.scene.set_pose(self.cam_node, pose=self.fix_pose(pose))
        
        return self.renderer.render(self.scene, flags=self.render_flags)

    def fix_pose(self, pose):
        # Corrects OpenCV (Z-forward) to OpenGL (Z-backward)
        R = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def delete(self):
        self.renderer.delete()

def render_depths(mesh, poses, intrinsics, H, W, nodata=0.0):
    mesh_opengl = pyrender.Mesh.from_trimesh(mesh)
    renderer = FastRenderer(mesh_opengl, height=H, width=W)
    
    rendered_depths = []
    try:
        for pose, K in tqdm(zip(poses, intrinsics), total=len(poses)):
            _, depth_pred = renderer.render_frame(H, W, K, pose)
            depth_pred = np.where(depth_pred == 0, nodata, depth_pred)
            rendered_depths.append(depth_pred)
    finally:
        renderer.delete()
    return rendered_depths

def render_normals(mesh, poses, intrinsics, H, W):
    # Extract vertex normals and normalize them to RGB space
    # Formula: RGB = (Normals + 1) / 2
    normals = mesh.vertex_normals
    rgb_normals = (normals + 1.0) / 2.0 
    rgb_normals_uint8 = (rgb_normals * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = rgb_normals_uint8

    # Create Pyrender mesh
    mesh_opengl = pyrender.Mesh.from_trimesh(mesh)
    renderer = FastRenderer(mesh_opengl, height=H, width=W)
    
    rendered_normals = []
    try:
        for pose, K in tqdm(zip(poses, intrinsics), total=len(poses)):
            normal_pred, depth_pred = renderer.render_frame(H, W, K, pose)
            normals_f32 = (normal_pred.astype(np.float32) / 127.5) - 1.0

            # Apply the mask to force those pixels to [0.0, 0.0, 0.0]
            mask = (depth_pred == 0)
            normals_f32[mask] = [0.0, 0.0, 0.0]

            rendered_normals.append(normals_f32)
    finally:
        renderer.delete()
    return rendered_normals

def render_DNs(mesh, poses, intrinsics, H, W, nodata=0.0):
    """
    Render both depths and normals
    """
    # Extract vertex normals and normalize them to RGB space
    # Formula: RGB = (Normals + 1) / 2
    normals = mesh.vertex_normals
    rgb_normals = (normals + 1.0) / 2.0 
    rgb_normals_uint8 = (rgb_normals * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = rgb_normals_uint8

    # Create Pyrender mesh
    mesh_opengl = pyrender.Mesh.from_trimesh(mesh)
    renderer = FastRenderer(mesh_opengl, height=H, width=W)
    
    rendered_depths = []
    rendered_normals = []
    try:
        for pose, K in tqdm(zip(poses, intrinsics), total=len(poses)):
            normal_pred, depth_pred = renderer.render_frame(H, W, K, pose)
            normals_f32 = (normal_pred.astype(np.float32) / 127.5) - 1.0

            # Apply the mask to force those pixels to [0.0, 0.0, 0.0]
            mask = (depth_pred == 0)
            normals_f32[mask] = [0.0, 0.0, 0.0]
            rendered_normals.append(normals_f32)

            depth_pred = np.where(depth_pred == 0, nodata, depth_pred)
            rendered_depths.append(depth_pred)
    finally:
        renderer.delete()
    return rendered_depths, rendered_normals
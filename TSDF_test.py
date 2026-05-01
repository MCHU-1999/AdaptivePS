from planarsplat.utils.mesh_util import refuse_mesh
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
from planarsplat.data_process.colmap_io import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
import open3d as o3d


def get_depth_normal_paths(depth_prior_path: str, normal_prior_path: str, img_name_list: list[str]):
    """Get paths to depth and normal files instead of loading them"""
    normal_paths = []
    depth_paths = []

    for img_name in img_name_list:
        filename = img_name.rsplit('.', 1)[0] + '.npy'

        # Get depth file path
        depth_file_path = os.path.join(depth_prior_path, filename)
        if os.path.exists(depth_file_path):
            depth_paths.append(depth_file_path)
        else:
            raise FileNotFoundError(f"Depth file not found: {depth_file_path}")

        # Get normal file path
        normal_file_path = os.path.join(normal_prior_path, filename)
        if os.path.exists(normal_file_path):
            normal_paths.append(normal_file_path)
        else:
            raise FileNotFoundError(f"Normal file not found: {normal_file_path}")

    return depth_paths, normal_paths

def read_DA3_data(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'The input data path {data_dir} does not exist.')
    else:
        depth_prior_path = os.path.join(data_dir, "DA3_depth")
        normal_prior_path = os.path.join(data_dir, "DA3_normal")
    
    colmap_cam_file_path = os.path.join(data_dir, 'DA3_colmap/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        colmap_cam_file_path = os.path.join(data_dir, 'DA3_colmap/cameras.txt')
        if not os.path.exists(colmap_cam_file_path):
            raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_dir, 'DA3_colmap/images.bin')
    if not os.path.exists(colmap_image_file_path):
        colmap_image_file_path = os.path.join(data_dir, 'DA3_colmap/images.txt')
        if not os.path.exists(colmap_image_file_path):
            raise ValueError(f'The input path {colmap_image_file_path} does not exist.')
    
    if colmap_cam_file_path.endswith(".bin"):
        cameras = read_intrinsics_binary(colmap_cam_file_path)
    else: 
        cameras = read_intrinsics_text(colmap_cam_file_path)

    if colmap_image_file_path.endswith(".bin"):
        images_meta = read_extrinsics_binary(colmap_image_file_path)
    else:
        images_meta = read_extrinsics_text(colmap_image_file_path)
    
    intrinsics_list = []
    c2ws_list = []
    img_name_list = []

    for img_id, img_meta in images_meta.items():
        # Extrinsics
        frame_name = img_meta.name

        q = img_meta.qvec
        t = img_meta.tvec
        r = qvec2rotmat(q)
        rt = np.eye(4)
        rt[:3,:3] = r
        rt[:3, 3] = t
        c2w = np.linalg.inv(rt).astype(np.float32)

        c2ws_list.append(c2w)
        img_name_list.append(frame_name)

        # Intrinsics
        camera = cameras[img_id]
        fx, fy, cx, cy = camera.params[:4]
        intrinsic = np.array([[fx, 0., cx],
                                [0., fy, cy],
                                [0., 0., 1.0]]).astype(np.float32)
        h = camera.height
        w = camera.width
        intrinsics_list.append(intrinsic)

    assert len(intrinsics_list) == len(c2ws_list), "Dataset generated from DA3 should have the same amount of images as cameras"

    # Fetch pre-computed, aligned depth and normal maps paths (lazy loading)
    depth_paths, normal_paths = get_depth_normal_paths(depth_prior_path, normal_prior_path, img_name_list)
    img_res = [h, w]

    data = {
        'depth_paths': depth_paths,  # Use corrected depth paths instead
        'normal_paths': normal_paths,          # Use normal paths instead of loaded data
        'extrinsics': c2ws_list,  # c2w
        'intrinsics': intrinsics_list,
        'img_res': img_res
    }
    return data

def read_prior_data(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'The input data path {data_dir} does not exist.')
    else:
        depth_prior_path = os.path.join(data_dir, "scaled_depth")
        normal_prior_path = os.path.join(data_dir, "mono_normal")
    
    colmap_cam_file_path = os.path.join(data_dir, 'sparse/0/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_dir, 'sparse/0/images.bin')
    if not os.path.exists(colmap_image_file_path):
        raise ValueError(f'The input path {colmap_image_file_path} does not exist.')


    cameras = read_intrinsics_binary(colmap_cam_file_path)
    camera = next(iter(cameras.values()))
    fx, fy, cx, cy = camera.params[:4]
    intrinsic = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.0]]).astype(np.float32)
    h = camera.height
    w = camera.width
    
    images_meta = read_extrinsics_binary(colmap_image_file_path)
    
    # color_images_list = []
    c2ws_list = []
    intrinsics_list = []
    img_name_list = []

    for img_id, img_meta in images_meta.items():
        frame_name = img_meta.name

        q = img_meta.qvec
        t = img_meta.tvec
        r = qvec2rotmat(q)
        rt = np.eye(4)
        rt[:3,:3] = r
        rt[:3, 3] = t
        c2w = np.linalg.inv(rt).astype(np.float32)

        c2ws_list.append(c2w)
        intrinsics_list.append(intrinsic)
        img_name_list.append(frame_name)

    # Fetch pre-computed, aligned depth and normal maps paths (lazy loading)
    depth_paths, normal_paths = get_depth_normal_paths(depth_prior_path, normal_prior_path, img_name_list)
    img_res = [h, w]

    data = {
        'depth_paths': depth_paths,  # Use corrected depth paths instead
        'normal_paths': normal_paths,          # Use normal paths instead of loaded data
        'extrinsics': c2ws_list,  # c2w
        'intrinsics': intrinsics_list,
        'img_res': img_res
    }
    return data

def fuse_a_mesh(data, depth_trunc, mesh_out_path):
    print("Loading depth maps for mesh generation...")
    mono_depths = [np.load(depth_path).astype(np.float32) for depth_path in data['depth_paths']]
    print(f"Loaded {len(mono_depths)} depth maps for mesh generation")

    img_res = data['img_res']
    h, w = img_res[0], img_res[1]

    mesh = refuse_mesh(
        [d.squeeze().reshape(h, w) for d in mono_depths],
        [np.asarray(x, dtype=np.float32) for x in data['extrinsics']],
        [np.asarray(x, dtype=np.float32) for x in data['intrinsics']],
        h,
        w,
        voxel_length=0.02,
        sdf_trunc=0.08,
        depth_trunc=depth_trunc,
    )

    o3d.io.write_triangle_mesh(mesh_out_path, mesh)


if __name__ == '__main__':

    # DEPTH_TRUNC_LIST = [5, 10, 15, 20, 25, 30]
    DEPTH_TRUNC_LIST = [5, 4, 3]

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_dir", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-o", "--out_dir", type=str, default='TSDF_test/test', help='path of output dir')

    args = parser.parse_args()
    data_dir = args.data_dir

    data_DA3 = read_DA3_data(data_dir)
    data_prior = read_prior_data(data_dir)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for depth_trunc in DEPTH_TRUNC_LIST:
        print(f"Fusing meshes (depth_trunc = {depth_trunc})")

        DA3_mesh_out_path = os.path.join(out_dir, f'DA3_trunc{depth_trunc}.ply')
        fuse_a_mesh(data_DA3, depth_trunc, DA3_mesh_out_path)

        mesh_out_path = os.path.join(out_dir, f'vanilla_trunc{depth_trunc}.ply')
        fuse_a_mesh(data_prior, depth_trunc, mesh_out_path)

        
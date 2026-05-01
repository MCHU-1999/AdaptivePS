import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
import numpy as np
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_planarSplatting import run_planarSplatting
from planarsplat.data_process.colmap_io import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from PIL import Image
from planarsplat.utils.misc_util import put_if_not_none
import cv2
import gc


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/blender', help='path of output dir')
    parser.add_argument("--conf_path", type=str, default='configs/blender.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')

    parser.add_argument("--mesh_pre_align", type=bool, default=False, help='Optional pre-alignment of depth maps')
    parser.add_argument("--voxel_length", type=float, default=None, help='voxel size for TSDF Integration')
    parser.add_argument("--depth_trunc", type=float, default=None, help='max meaningful depth in TSDF (a threshold that separates target and background)')
    parser.add_argument("--max_depth", type=float, default=None, help='max meaningful depth in loss function (a threshold that separates target and background)')
    parser.add_argument("--exp_name", type=str, default=None, help='experiment name for output folder')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    else:
        depth_prior_path = os.path.join(data_path, "blender_depth")
        normal_prior_path = os.path.join(data_path, "blender_normal")
    
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        raise ValueError(f'The input image path {image_path} does not exist.')

    colmap_cam_file_path = os.path.join(data_path, 'sparse/0/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_path, 'sparse/0/images.bin')
    if not os.path.exists(colmap_image_file_path):
        raise ValueError(f'The input path {colmap_image_file_path} does not exist.')


    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(data_path, 'blender_precomputed.pth')
    use_precomputed_data = args.use_precomputed_data

    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        cameras = read_intrinsics_binary(colmap_cam_file_path)
        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        intrinsic = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.0]]).astype(np.float32)
        h = camera.height
        w = camera.width
        
        images_meta = read_extrinsics_binary(colmap_image_file_path)
        
        image_paths_list = []
        c2ws_list = []
        intrinsics_list = []
        img_id_list = []
        img_name_list = []

        i = 0
        for img_id, img_meta in images_meta.items():
            frame_name = img_meta.name
            frame_path = os.path.join(image_path, frame_name)

            q = img_meta.qvec
            t = img_meta.tvec
            r = qvec2rotmat(q)
            rt = np.eye(4)
            rt[:3,:3] = r
            rt[:3, 3] = t
            c2w = np.linalg.inv(rt).astype(np.float32)

            c2ws_list.append(c2w)
            intrinsics_list.append(intrinsic)
            image_paths_list.append(frame_path)
            img_id_list.append(img_id)
            img_name_list.append(frame_name)

        # Fetch pre-computed, aligned depth and normal maps paths (lazy loading)
        depth_paths, normal_paths = get_depth_normal_paths(depth_prior_path, normal_prior_path, img_name_list)
        img_res = [h, w]

        data = {
            'depth_paths': depth_paths,  # Store paths instead of loaded data
            'normal_paths': normal_paths,  # Store paths instead of loaded data
            'image_paths': image_paths_list,
            'extrinsics': c2ws_list,  # c2w
            'intrinsics': intrinsics_list,
            # 'out_path': out_path
        }
        torch.save(data, precomputed_data_path)

    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/base_confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    put_if_not_none(conf, 'train.exps_folder_name', out_path)

    if use_precomputed_data and 'img_res' not in locals():
        with Image.open(data['image_paths'][0]) as img:
            img_res = [img.height, img.width]
    
    voxel_length = args.voxel_length
    sdf_trunc = voxel_length * 4 if voxel_length else None
    put_if_not_none(conf, 'dataset.voxel_length', voxel_length)
    put_if_not_none(conf, 'dataset.sdf_trunc', sdf_trunc)
    put_if_not_none(conf, 'dataset.max_depth', args.max_depth)
    put_if_not_none(conf, 'dataset.depth_trunc', args.depth_trunc)
    put_if_not_none(conf, 'train.expname', args.exp_name)
    put_if_not_none(conf, 'dataset.img_res', img_res)
    put_if_not_none(conf, 'dataset.mesh_pre_align', args.mesh_pre_align)

    planar_rec = run_planarSplatting(data=data, conf=conf)
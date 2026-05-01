import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
import numpy as np
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_metric3d import extract_mono_geo_demo, predict_mono_geo_demo, predict_masked_mono_geo
from utils_demo.run_planarSplatting import run_planarSplatting
from utils_demo.read_write_model import read_model
from planarsplat.data_process.colmap_io import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
from planarsplat.utils.misc_util import put_if_not_none
from PIL import Image
import cv2


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

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/make_depth_scale.py
def get_scales(key, cameras, images, points3d_ordered, invmonodepthmap):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_meta[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1

    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]

        # Debug prints to identify which dimension is too large
        print(f"invmonodepthmap shape: {invmonodepthmap.shape}")
        print(f"maps shape: {maps.shape}")
        # print(f"maps min/max: x=[{maps[..., 0].min():.1f}, {maps[..., 0].max():.1f}], y=[{maps[..., 1].min():.1f}, {maps[..., 1].max():.1f}]")
        if invmonodepthmap.shape[0] >= 32767 or invmonodepthmap.shape[1] >= 32767:
            print(f"ERROR: invmonodepthmap dimensions too large: {invmonodepthmap.shape}")
        if maps.shape[0] >= 32767:
            print(f"ERROR: maps has too many points: {maps.shape[0]}")

        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/DA3FG', help='path of output dir')
    parser.add_argument("--conf_path", type=str, default='configs/DA3FG.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    parser.add_argument('--mask', type=str, default=None, help='name of mask folder (None=not using mask)')

    parser.add_argument("--mesh_pre_align", type=bool, default=False, help='Optional pre-alignment of depth maps')
    parser.add_argument("--voxel_length", type=float, default=None, help='voxel size for TSDF Integration')
    parser.add_argument("--max_depth", type=float, default=None, help='max meaningful depth in loss function (a threshold that separates target and background)')
    parser.add_argument("--exp_name", type=str, default=None, help='experiment name for output folder')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    else:
        depth_prior_path = os.path.join(data_path, "DA3_depth")
        normal_prior_path = os.path.join(data_path, "DA3_normal")
    
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        raise ValueError(f'The input image path {image_path} does not exist.')

    colmap_cam_file_path = os.path.join(data_path, 'DA3_colmap/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        colmap_cam_file_path = os.path.join(data_path, 'DA3_colmap/cameras.txt')
        if not os.path.exists(colmap_cam_file_path):
            raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_path, 'DA3_colmap/images.bin')
    if not os.path.exists(colmap_image_file_path):
        colmap_image_file_path = os.path.join(data_path, 'DA3_colmap/images.txt')
        if not os.path.exists(colmap_image_file_path):
            raise ValueError(f'The input path {colmap_image_file_path} does not exist.')
    
    USE_MASK = False
    if args.mask is not None:
        mask_folder_name = args.mask
        USE_MASK = True
        fg_mask_dir = os.path.join(data_path, mask_folder_name)
        if not os.path.exists(fg_mask_dir):
            raise ValueError(f'The mask directory {fg_mask_dir} does not exist.')
    else:
        raise NotImplementedError(f'Guess what, you have to have masks for this:)')


    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(data_path, 'DA3FG_precomputed.pth')
    use_precomputed_data = args.use_precomputed_data

    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        if colmap_cam_file_path.endswith(".bin"):
            cameras = read_intrinsics_binary(colmap_cam_file_path)
        else: 
            cameras = read_intrinsics_text(colmap_cam_file_path)

        if colmap_image_file_path.endswith(".bin"):
            images_meta = read_extrinsics_binary(colmap_image_file_path)
        else:
            images_meta = read_extrinsics_text(colmap_image_file_path)
        
        intrinsics_list = []
        image_paths_list = []
        c2ws_list = []
        img_id_list = []
        img_name_list = []
        fg_mask_paths = []

        for img_id, img_meta in images_meta.items():
            # Extrinsics
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
            image_paths_list.append(frame_path)
            img_id_list.append(img_id)
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

            if USE_MASK:
                fg_mask_path = os.path.join(fg_mask_dir, frame_name)
                fg_mask_paths.append(fg_mask_path)

        assert len(intrinsics_list) == len(c2ws_list), "Dataset generated from DA3 should have the same amount of images as cameras"

        # Fetch pre-computed, aligned depth and normal maps paths (lazy loading)
        depth_paths, normal_paths = get_depth_normal_paths(depth_prior_path, normal_prior_path, img_name_list)
        img_res = [h, w]

        data = {
            'image_paths': image_paths_list,
            'depth_paths': depth_paths,  # Use corrected depth paths instead
            'normal_paths': normal_paths,          # Use normal paths instead of loaded data
            'extrinsics': c2ws_list,  # c2w
            'intrinsics': intrinsics_list
        }
        if USE_MASK:
            data['fg_mask_paths'] = fg_mask_paths

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
    put_if_not_none(conf, 'train.expname', args.exp_name)
    put_if_not_none(conf, 'dataset.img_res', img_res)
    put_if_not_none(conf, 'dataset.mesh_pre_align', args.mesh_pre_align)

    planar_rec = run_planarSplatting(data=data, conf=conf)
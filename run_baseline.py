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
from planarsplat.utils.timing_util import Timer, save_runtime_json
from PIL import Image
import cv2

RUNTIME_LOG_PATH = "evaluation/runtime_logs/baseline.json"


# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/make_depth_scale.py
def get_scales(key, cameras, images, points3d_ordered, invmonodepthmap):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # pts_idx = images_meta[key].point3D_ids
    pts_idx = image_meta.point3D_ids

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


def run_baseline(data_path, out_path, conf_path, use_precomputed_data=False, mask=None, mesh_pre_align=False, voxel_length=None, depth_trunc=None, max_depth=None, exp_name=None):
    scene_name = exp_name or os.path.basename(data_path.rstrip('/'))
    with Timer() as t_total:
        _run_baseline(
            data_path=data_path, out_path=out_path, conf_path=conf_path,
            use_precomputed_data=use_precomputed_data, mask=mask,
            mesh_pre_align=mesh_pre_align, voxel_length=voxel_length,
            depth_trunc=depth_trunc, max_depth=max_depth, exp_name=exp_name,
            scene_name=scene_name,
        )
    save_runtime_json(RUNTIME_LOG_PATH, {scene_name: {"total_s": round(t_total.elapsed, 2)}})


def _run_baseline(data_path, out_path, conf_path, use_precomputed_data=False, mask=None, mesh_pre_align=False, voxel_length=None, depth_trunc=None, max_depth=None, exp_name=None, scene_name=None):
    USE_MASK = False
    if mask is not None:
        mask_folder_name = mask
        USE_MASK = True

    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        raise ValueError(f'The input image path {image_path} does not exist.')
    
    if USE_MASK:
        fg_mask_dir = os.path.join(data_path, mask_folder_name)
        if not os.path.exists(fg_mask_dir):
            raise ValueError(f'The mask directory {fg_mask_dir} does not exist.')

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
    
    depth_save_dir = os.path.join(data_path, 'mono_depth')
    normal_save_dir = os.path.join(data_path, 'mono_normal')
    scaled_depth_dir = os.path.join(data_path, 'scaled_depth')
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(normal_save_dir, exist_ok=True)
    os.makedirs(scaled_depth_dir, exist_ok=True)

    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(data_path, 'baseline_precomputed.pth')

    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        if colmap_cam_file_path.endswith(".bin"):
            cameras = read_intrinsics_binary(colmap_cam_file_path)
        else: 
            cameras = read_intrinsics_text(colmap_cam_file_path)

        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        intrinsic = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.0]]).astype(np.float32)
        h = camera.height
        w = camera.width
        
        if colmap_image_file_path.endswith(".bin"):
            images_meta = read_extrinsics_binary(colmap_image_file_path)
        else:
            images_meta = read_extrinsics_text(colmap_image_file_path)
        
        color_images_list = []
        fg_masks_list = []
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
            rgb = np.array(Image.open(frame_path))  # h, w, 3

            c2ws_list.append(c2w)
            intrinsics_list.append(intrinsic)
            image_paths_list.append(frame_path)
            color_images_list.append(rgb)
            img_id_list.append(img_id)
            img_name_list.append(frame_name)

            if USE_MASK:
                fg_mask_path = os.path.join(fg_mask_dir, frame_name)
                fg_mask = np.array(Image.open(fg_mask_path))  # h, w, 1
                fg_mask = (fg_mask > 0.5).astype(int)
                fg_masks_list.append(np.squeeze(fg_mask))

        # run metric3dv2
        with Timer() as t_m3d:
            if USE_MASK:
                depth_paths, normal_paths = predict_masked_mono_geo(
                    img_name_list, color_images_list, fg_masks_list, intrinsics_list, depth_save_dir, normal_save_dir)
            else:
                depth_paths, normal_paths = predict_mono_geo_demo(
                    img_name_list, color_images_list, intrinsics_list, depth_save_dir, normal_save_dir)
        save_runtime_json(RUNTIME_LOG_PATH, {scene_name: {"metric3dv2_s": round(t_m3d.elapsed, 2)}})
        
        img_res = [h, w]
        del color_images_list

        # cam_intrinsics, images_metas, points3d = read_model(os.path.join(data_path, "sparse", "0"))
        cam_intrinsics, images_metas, points3d = read_model(os.path.join(data_path, "DA3_colmap"))
        pts_indices = np.array([points3d[key].id for key in points3d])
        pts_xyzs = np.array([points3d[key].xyz for key in points3d])
        points3d_ordered = np.zeros([pts_indices.max()+1, 3])
        points3d_ordered[pts_indices] = pts_xyzs
        
        scaled_depth_paths = []
        print("Applying scale correction to depth maps...")
        for i, (img_name, img_id, depth_path) in enumerate(zip(img_name_list, img_id_list, depth_paths)):
            # Load depth map temporarily
            monodepth = np.load(depth_path)
            
            # Apply scale correction (same as before)
            invmonodepthmap = 1 / monodepth
            res = get_scales(img_id, cam_intrinsics, images_metas, points3d_ordered, invmonodepthmap)
            scale = res['scale']
            offset = res['offset']
            print(f"Image {img_name}: scale={scale:.4f}, offset={offset:.4f}")  # Add this debug line
            if scale > 0:
                invmonodepthmap = invmonodepthmap * scale + offset
                monodepth = 1 / invmonodepthmap
                monodepth = np.clip(monodepth, 0, 300)  # Add clipping after scaling
            else:
                print(f"Warning: Invalid scale for image {img_name}, using original depth")  # Add this warning
            
            if USE_MASK:
                monodepth *= fg_masks_list[i]

            # Save corrected depth map
            img_name = img_name.rsplit('.', 1)[0]
            scaled_depth_path = os.path.join(scaled_depth_dir, f'{img_name}.npy')
            np.save(scaled_depth_path, monodepth.astype(np.float32))
            scaled_depth_paths.append(scaled_depth_path)

        data = {
            'image_paths': image_paths_list,
            'depth_paths': scaled_depth_paths,  # Use corrected depth paths instead
            'normal_paths': normal_paths,          # Use normal paths instead of loaded data
            'extrinsics': c2ws_list,  # c2w
            'intrinsics': intrinsics_list,
            # 'out_path': out_path
        }
        if USE_MASK:
            del fg_masks_list
        torch.save(data, precomputed_data_path)

    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/base_confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    put_if_not_none(conf, 'train.exps_folder_name', out_path)

    if use_precomputed_data and 'img_res' not in locals():
        with Image.open(data['image_paths'][0]) as img:
            img_res = [img.height, img.width]
    
    sdf_trunc = voxel_length * 4 if voxel_length else None
    put_if_not_none(conf, 'dataset.voxel_length', voxel_length)
    put_if_not_none(conf, 'dataset.sdf_trunc', sdf_trunc)
    put_if_not_none(conf, 'dataset.max_depth', max_depth)
    put_if_not_none(conf, 'dataset.depth_trunc', depth_trunc)
    put_if_not_none(conf, 'train.expname', exp_name)
    put_if_not_none(conf, 'dataset.img_res', img_res)
    put_if_not_none(conf, 'dataset.mesh_pre_align', mesh_pre_align)

    planar_rec = run_planarSplatting(data=data, conf=conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/baseline', help='path of output dir')
    parser.add_argument("--conf_path", type=str, default='configs/baseline.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    parser.add_argument('--mask', type=str, default=None, help='name of mask folder (None=not using mask)')

    parser.add_argument("--mesh_pre_align", type=bool, default=False, help='Optional pre-alignment of depth maps')
    parser.add_argument("--voxel_length", type=float, default=None, help='voxel size for TSDF Integration')
    parser.add_argument("--depth_trunc", type=float, default=None, help='max meaningful depth in TSDF (a threshold that separates target and background)')
    parser.add_argument("--max_depth", type=float, default=None, help='max meaningful depth in loss function (a threshold that separates target and background)')
    parser.add_argument("--exp_name", type=str, default=None, help='experiment name for output folder')
    args = parser.parse_args()

    run_baseline(
        data_path=args.data_path,
        out_path=args.out_path,
        conf_path=args.conf_path,
        use_precomputed_data=args.use_precomputed_data,
        mask=args.mask,
        mesh_pre_align=args.mesh_pre_align,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
        max_depth=args.max_depth,
        exp_name=args.exp_name
    )
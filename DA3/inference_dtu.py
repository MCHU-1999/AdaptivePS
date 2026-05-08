import glob, torch, sys, os
import numpy as np
from PIL import Image
import math
from typing import NamedTuple, List, Dict
from depth_anything_3.api import DepthAnything3
from loguru import logger

from typing import NamedTuple, List, Dict


NPZ_PATH = os.path.join(os.path.dirname(__file__), "dtu_cameras.npz")


class DtuDataset(NamedTuple):
    extrinsics_list: np.ndarray     # (N, 4, 4)
    intrinsics_list: np.ndarray     # (N, 3, 3)
    img_paths_list: np.ndarray | List[str]
    img_name_list: np.ndarray | List[str]
    width: int = 1554
    height: int = 1162
    N: int = 49

def read_dtu_dataset(data_dir, camera_npz_path):
    ##
    # The important checks
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f'The input path {image_dir} does not exist.')
    
    if not os.path.exists(camera_npz_path):
        raise ValueError(f'The input path {camera_npz_path} does not exist.')
    
    ##
    # Read intrinsics and extrinsics
    camera_params = np.load(camera_npz_path)
    intrinsic = camera_params['K']
    extrinsics_list = camera_params['extrinsics']
    
    # Read and sort images from directory based on int(filename)
    img_files = glob.glob(os.path.join(image_dir, "*.png"))
    img_files = sorted(img_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    img_paths_list = img_files
    img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    
    # Create intrinsics list by repeating intrinsic matrix for each image
    num_images = len(img_files)
    intrinsics_list = [intrinsic for _ in range(num_images)]

    dataset = DtuDataset(
        extrinsics_list=np.stack(extrinsics_list, axis=0),
        intrinsics_list=np.stack(intrinsics_list, axis=0),
        img_paths_list=img_paths_list,
        img_name_list=img_name_list,
        N=num_images
    )
    
    return dataset

def da3_inference_a_scene(scene):
    logger.info(f"DA3 Inference on scene: {scene['exp_name']}")
    data_dir = scene["data_path"]

    # Setup model and device
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    dataset = read_dtu_dataset(data_dir, NPZ_PATH)

    # Export depth data and 3D visualization
    prediction = model.inference(
        image=dataset.img_paths_list,
        extrinsics=dataset.extrinsics_list,
        intrinsics=dataset.intrinsics_list,
        export_dir=data_dir,
        export_format="planarsplatting-colmap",
        process_res=420,
        # process_res=840,
        process_res_method="upper_bound_resize",
        export_kwargs={
            "planarsplatting": {
                "img_name_list": dataset.img_name_list,
                "img_res": [dataset.height, dataset.width]
            }
        },
        show_cameras=False,
        conf_thresh_percentile=40,
        num_max_points=100_000,
    )

    # # FOR DEBUG
    # # prediction.processed_images : [N, H, W, 3] uint8   array
    # logger.info(f"processed_images.shape: {prediction.processed_images.shape}")
    # # prediction.depth            : [N, H, W]    float32 array
    # logger.info(f"depth.shape: {prediction.depth.shape}")  
    # # prediction.conf             : [N, H, W]    float32 array
    # logger.info(f"conf.shape: {prediction.conf.shape}")  
    # # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    # logger.info(f"extrinsics.shape: {prediction.extrinsics.shape}")
    # # prediction.intrinsics       : [N, 3, 3]    float32 array
    # logger.info(f"intrinsics.shape: {prediction.intrinsics.shape}")

    return None

def da3_inference_all_scenes(scenes):
    # Setup model and device
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    for scene in scenes:
        logger.info(f"DA3 Inference on scene: {scene['exp_name']}")
        data_dir = scene["data_path"]
        dataset = read_dtu_dataset(data_dir, NPZ_PATH)

        # Export depth data and 3D visualization
        prediction = model.inference(
            image=dataset.img_paths_list,
            extrinsics=dataset.extrinsics_list,
            intrinsics=dataset.intrinsics_list,
            export_dir=data_dir,
            export_format="planarsplatting-colmap",
            process_res=420,
            # process_res=840,
            process_res_method="upper_bound_resize",
            export_kwargs={
                "planarsplatting": {
                    "img_name_list": dataset.img_name_list,
                    "img_res": [dataset.height, dataset.width]
                }
            },
            show_cameras=False,
            conf_thresh_percentile=40,
            num_max_points=100_000,
        )

    return None
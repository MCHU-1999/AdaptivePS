import torch, sys, os
import numpy as np
from PIL import Image
import math
from typing import NamedTuple, List, Dict
from depth_anything_3.api import DepthAnything3
from loguru import logger


class Dataset(NamedTuple):
    img_paths_list: np.ndarray | List[str]
    img_name_list: np.ndarray | List[str]
    bldg_mask_paths: np.ndarray | List[str]
    gnd_mask_paths: np.ndarray | List[str]
    width: int
    height: int
    N: int

def synthesize_intrinsics(
    width: int,
    height: int,
    fov_deg: float = 75,
) -> np.ndarray:
    """
    Synthesize camera intrinsic matrix from field of view.
    
    The FOV is applied to the longer dimension (width for landscape, height for portrait).
    Square pixels (fx == fy) are assumed.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Field of view in degrees applied to the longer side (default: 75)
    
    Returns:
        K: Camera intrinsic matrix (3, 3) as float32 numpy array
        
    Example:
        K = synthesize_intrinsics(1280, 720, fov_deg=75)  # landscape: hfov=75
        K = synthesize_intrinsics(720, 1280, fov_deg=75)  # portrait: vfov=75
    """
    cx = width / 2.0
    cy = height / 2.0
    
    if width >= height:
        # Landscape or square: apply FOV to width (horizontal FOV)
        fov_rad = math.radians(fov_deg)
        f = (width / 2.0) / math.tan(fov_rad / 2.0)
    else:
        # Portrait: apply FOV to height (vertical FOV)
        fov_rad = math.radians(fov_deg)
        f = (height / 2.0) / math.tan(fov_rad / 2.0)
    
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return K

def read_dataset(data_dir):
    ##
    # The important checks
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f'The input path {image_dir} does not exist.')
    bldg_masks_dir = os.path.join(data_dir, "bldg_masks")
    if not os.path.exists(bldg_masks_dir):
        raise ValueError(f'The input path {bldg_masks_dir} does not exist.')
    gnd_masks_dir = os.path.join(data_dir, "gnd_masks")
    if not os.path.exists(gnd_masks_dir):
        raise ValueError(f'The input path {gnd_masks_dir} does not exist.')

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
    )

    if len(image_files) == 0:
        raise ValueError(f"No valid images found in {image_dir}")

    img_paths_list = [os.path.join(image_dir, f) for f in image_files]
    img_name_list = [f.rsplit('.', 1)[0] for f in image_files]
    bldg_mask_paths = [os.path.join(bldg_masks_dir, f) for f in image_files]
    gnd_masks_paths = [os.path.join(gnd_masks_dir, f) for f in image_files]

    with Image.open(img_paths_list[0]) as img:
        w, h = img.size

    for img_path in img_paths_list[1:]:
        with Image.open(img_path) as img:
            cur_w, cur_h = img.size
        if cur_w != w or cur_h != h:
            raise ValueError(
                f"Mixed image resolutions are not supported: expected {w}x{h}, got {cur_w}x{cur_h} for {img_path}"
            )

    dataset = Dataset(
        img_paths_list=img_paths_list,
        img_name_list=img_name_list,
        bldg_mask_paths=bldg_mask_paths,
        gnd_mask_paths=gnd_masks_paths,
        width=w,
        height=h,
        N=len(img_paths_list)
    )
    
    return dataset

def da3_inference_a_scene(scene):
    logger.info(f"DA3 Inference on scene: {scene['exp_name']}")
    data_dir = scene["data_path"]

    # Setup model and device
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    dataset = read_dataset(data_dir)

    # Create synthetic intrinsics for outdoor scene (75° horizontal FOV)
    K_synthetic = synthesize_intrinsics(dataset.width, dataset.height, fov_deg=75)
    intrinsics = np.stack([K_synthetic] * dataset.N, axis=0)

    # Export depth data and 3D visualization
    prediction = model.inference(
        image=dataset.img_paths_list,
        extrinsics=None,
        intrinsics=intrinsics,
        export_dir=data_dir,
        export_format="planarsplatting-colmap-bldg_pts",
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
        num_max_points=100000,
        bldg_mask_paths=dataset.bldg_mask_paths,
        gnd_mask_paths=dataset.gnd_mask_paths
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
        dataset = read_dataset(data_dir)

        # Create synthetic intrinsics for outdoor scene (75° horizontal FOV)
        K_synthetic = synthesize_intrinsics(dataset.width, dataset.height, fov_deg=75)
        intrinsics = np.stack([K_synthetic] * dataset.N, axis=0)

        # Export depth data and 3D visualization
        prediction = model.inference(
            image=dataset.img_paths_list,
            extrinsics=None,
            intrinsics=intrinsics,
            export_dir=data_dir,
            export_format="planarsplatting-colmap-bldg_pts",
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
            num_max_points=100000,
            bldg_mask_paths=dataset.bldg_mask_paths,
            gnd_mask_paths=dataset.gnd_mask_paths
        )

    return None
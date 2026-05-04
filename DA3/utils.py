import os
import numpy as np
from PIL import Image
import math
from typing import NamedTuple, List, Dict

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
    

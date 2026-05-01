import os
import numpy as np
from PIL import Image
from inference import run_sequence_demo
from loguru import logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def set_hf_token_from_txt(filepath="./hf_token.txt"):
    # Read the token manually
    with open(filepath, "r") as f:
        token = f.read().strip()

    # Set it so the sam3 builder can find it
    os.environ["HF_TOKEN"] = token
    logger.info("HF_TOKEN set.")

    return None

def list_sorted_frames(data_dir):
    frame_files = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full_path) and ext in IMAGE_EXTS:
            frame_files.append(name)

    frame_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return frame_files

def to_numpy_mask(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    return mask_np

def save_masks_by_frame_index(outputs_per_frame, frame_dir, output_root_dir, mask_dir_name):
    frame_files = list_sorted_frames(frame_dir)
    assert len(frame_files) == len(outputs_per_frame), f"Amount of files ({len(frame_files)}) and masks ({len(outputs_per_frame)}) inconsistent."

    out_dir = os.path.join(output_root_dir, mask_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Get mask shape
    sample_path = os.path.join(frame_dir, frame_files[0])
    with Image.open(sample_path) as img:
        img_res = (img.height, img.width)

    saved = 0
    # .items() preserves insertion order in Python 3.7+
    for i, (frame_idx, obj_dict) in enumerate(outputs_per_frame.items()):
        
        if obj_dict:
            # mask_stack = np.stack([to_numpy_mask(mask) for mask in obj_dict.values()], axis=0)
            mask_stack = np.stack(list(obj_dict.values()), axis=0)
            combined_mask = np.any(mask_stack > 0, axis=0)
        else:
            logger.warning("Cannot find obj_dict, exporting all 0 masks.")
            combined_mask = np.zeros(img_res, dtype=bool)

        # Map the current iteration to the filename
        out_name = frame_files[i]
        out_path = os.path.join(out_dir, out_name)

        Image.fromarray((combined_mask.astype(np.uint8) * 255), mode="L").save(out_path)
        saved += 1

    return saved, out_dir

def inference_a_scene(scenes, mask_dir_name):
    for scene in scenes:
        frame_dir = f"{scene['data_path']}/images"
        outputs_per_frame = run_sequence_demo(frame_dir, scene['prompt'])
        saved, out_dir = save_masks_by_frame_index(
            outputs_per_frame,
            frame_dir,
            scene['data_path'],
            mask_dir_name,
        )
        logger.info(f"{scene['exp_name']}: saved {saved} masks to {out_dir}")

def inference_bd_gnd(scenes):
    for scene in scenes:
        logger.info(f"\nInference on scene: {scene['exp_name']}")
        frame_dir = f"{scene['data_path']}/images"

        # Building masks
        outputs_per_frame = run_sequence_demo(frame_dir, scene['bldg_prompt'])
        saved, out_dir = save_masks_by_frame_index(
            outputs_per_frame,
            frame_dir,
            scene['data_path'],
            'bldg_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} masks to {out_dir}")

        # Ground masks
        outputs_per_frame = run_sequence_demo(frame_dir, scene['gnd_prompt'])
        saved, out_dir = save_masks_by_frame_index(
            outputs_per_frame,
            frame_dir,
            scene['data_path'],
            'gnd_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} masks to {out_dir}")

# Main Function
if __name__ == "__main__":
    # CONST
    MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
    MASK_DIR_NAME = "fg_masks"
    SCENES = [
        # {
        #     "exp_name": "church-cadeby",
        #     "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "church-chesterfield",
        #     "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "clocktower",
        #     "data_path": f"{MY_STORAGE}/Pexels/clocktower",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "killingbeck-cemetery",
        #     "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "moskee-haarlem",
        #     "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "tower-court",
        #     "data_path": f"{MY_STORAGE}/Pexels/tower-court",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        # {
        #     "exp_name": "wotrubakirche",
        #     "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
        #     "prompt": "building in the front center",
        #     "bldg_prompt": "that building in the center of frame",
        #     "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        # },
        {
            "exp_name": "yorkshire-post",
            "data_path": f"{MY_STORAGE}/Pexels/yorkshire-post",
            "prompt": "building in the front center",
            "bldg_prompt": "that YORKSHIRE POST pole in the center of frame",
            "gnd_prompt": "every ground surfaces (sand, earth, grass, water, road, pavement)"
        }
    ]

    if not os.environ.get("HF_TOKEN"):
        token_path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
        if os.path.exists(token_path):
            set_hf_token_from_txt(token_path)
        else:
            raise FileNotFoundError(
                f"HF_TOKEN is not set and token file was not found: {token_path}"
            )

    inference_bd_gnd(SCENES)
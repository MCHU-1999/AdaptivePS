import argparse
import os
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path
from loguru import logger

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def run_sequence_demo(resource_path: str, prompt):
    """
    Run sequence/video SAM3 inference.

    resource_path can be:
    - a folder of JPEG frames
    - an MP4 file

    `prompt` can be a single string or a list of strings. When multiple prompts
    are provided, each prompt will be added to the session (useful to select
    different objects within the same mask pass).
    """
    predictor = build_sam3_video_predictor()

    start_response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": resource_path,
        }
    )
    session_id = start_response["session_id"]

    # accept either a single prompt string or an iterable of prompt strings
    prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
    for p in prompts:
        _ = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": p,
            }
        )

    outputs_per_frame = propagate_in_video(predictor, session_id)
    prompt_str = prompt if isinstance(prompt, str) else ",".join(prompts)
    print(f"[sequence] path={resource_path} prompt={prompt_str} -> {len(outputs_per_frame)} frames")
    
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # finally, close the inference session to free its GPU resources
    # (you may start a new session on another video)
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    return outputs_per_frame


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
    no_mask = 0
    # .items() preserves insertion order in Python 3.7+
    for i, (frame_idx, obj_dict) in enumerate(outputs_per_frame.items()):
        if obj_dict:
            # mask_stack = np.stack([to_numpy_mask(mask) for mask in obj_dict.values()], axis=0)
            mask_stack = np.stack(list(obj_dict.values()), axis=0)
            combined_mask = np.any(mask_stack > 0, axis=0)
        else:
            # logger.warning("Cannot find obj_dict, exporting all 1 masks.")
            combined_mask = np.ones(img_res, dtype=bool)
            no_mask += 1

        # Map the current iteration to the filename
        out_name = frame_files[i]
        out_path = os.path.join(out_dir, out_name)

        Image.fromarray((combined_mask.astype(np.uint8) * 255), mode="L").save(out_path)
        saved += 1

    return saved, no_mask, out_dir

def sam_inference_a_scene(scene):
    logger.info(f"\nInference on scene: {scene['exp_name']}")
    frame_dir = f"{scene['data_path']}/images"

    # Building masks
    outputs_per_frame = run_sequence_demo(frame_dir, scene['bldg_prompt'])
    saved, no_mask, out_dir = save_masks_by_frame_index(
        outputs_per_frame,
        frame_dir,
        scene['data_path'],
        'bldg_masks',
    )
    logger.info(f"{scene['exp_name']}: saved {saved} masks, {no_mask} of them are empty")

    # Ground masks
    outputs_per_frame = run_sequence_demo(frame_dir, scene['gnd_prompt'])
    saved, no_mask, out_dir = save_masks_by_frame_index(
        outputs_per_frame,
        frame_dir,
        scene['data_path'],
        'gnd_masks',
    )
    logger.info(f"{scene['exp_name']}: saved {saved} masks, {no_mask} of them are empty")

def inference_bd_gnd(scenes):
    for scene in scenes:
        logger.info(f"\nInference on scene: {scene['exp_name']}")
        frame_dir = f"{scene['data_path']}/images"

        # Building masks
        outputs_per_frame = run_sequence_demo(frame_dir, scene['bldg_prompt'])
        saved, no_mask, out_dir = save_masks_by_frame_index(
            outputs_per_frame,
            frame_dir,
            scene['data_path'],
            'bldg_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} masks, {no_mask} of them are empty")

        # Ground masks
        outputs_per_frame = run_sequence_demo(frame_dir, scene['gnd_prompt'])
        saved, no_mask, out_dir = save_masks_by_frame_index(
            outputs_per_frame,
            frame_dir,
            scene['data_path'],
            'gnd_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} masks, {no_mask} of them are empty")
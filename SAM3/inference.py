import os
from PIL import Image
import numpy as np
from loguru import logger
import torch
from sam3.model_builder import build_sam3_video_predictor, build_sam3_multiplex_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

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

def save_masks_by_frame_index(combined_mask_per_frame, frame_dir, output_root_dir, mask_dir_name):
    frame_files = list_sorted_frames(frame_dir)
    assert len(frame_files) == len(combined_mask_per_frame), f"Amount of files ({len(frame_files)}) and masks ({len(combined_mask_per_frame)}) inconsistent."

    out_dir = os.path.join(output_root_dir, mask_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Get mask shape
    sample_path = os.path.join(frame_dir, frame_files[0])
    with Image.open(sample_path) as img:
        img_res = (img.height, img.width)

    saved = 0
    no_mask = 0
    # .items() preserves insertion order in Python 3.7+
    for i, (frame_idx, mask) in enumerate(combined_mask_per_frame.items()):
        if mask is not None:
            # mask exist
            combined_mask = mask
        else:
            # logger.warning("Cannot find mask, exporting all 1 masks.")
            combined_mask = np.ones(img_res, dtype=bool)
            no_mask += 1

        # Map the current iteration to the filename
        out_name = frame_files[i]
        out_path = os.path.join(out_dir, out_name)

        Image.fromarray((combined_mask.astype(np.uint8) * 255), mode="L").save(out_path)
        saved += 1

    return saved, no_mask, out_dir


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def inference_bldg_video(predictor, scene):
    frame_dir = f"{scene['data_path']}/images"
    bldg_prompt = scene['bldg_prompt']

    # For bldg_mask we use video predictor bc the target is always there,
    # and there should only be one target building.
    start_response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": frame_dir,
        }
    )
    session_id = start_response["session_id"]

    # accept either a single prompt string or an iterable of prompt strings
    response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": bldg_prompt,
        }
    )
    # The 1st propagation
    initial_outputs = propagate_in_video(predictor, session_id)

    # Here we should remove the smaller building in background before propagation
    for frame_idx, outputs in initial_outputs.items():
        masks = outputs.get("out_binary_masks")
        if len(masks) > 0:
            counts = {}
            for idx, binary_mask in enumerate(masks):
                obj_id = outputs["out_obj_ids"][idx]
                counts[obj_id] = int((binary_mask > 0).sum())

            largest_obj_id = max(counts, key=counts.get)

            for obj_id in counts.keys():
                if obj_id != largest_obj_id:
                    predictor.handle_request(
                        request=dict(
                            type="remove_object",
                            session_id=session_id,
                            obj_id=obj_id,
                        )
                    )

    # The 2nd propagation
    outputs_per_frame = propagate_in_video(predictor, session_id)
    combined_mask_per_frame = {}
    for frame_idx, outputs in outputs_per_frame.items():
        masks = outputs.get("out_binary_masks")
        # Merge outputs from this prompt with existing frame outputs
        if len(masks) > 0:
            mask_stack = np.stack(masks, axis=0)
            combined_mask = np.any(mask_stack > 0, axis=0)
            combined_mask_per_frame[frame_idx] = combined_mask
        else: 
            combined_mask_per_frame[frame_idx] = None
    
    # finally, close the inference session to free its GPU resources
    # (you may start a new session on another video)
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    return combined_mask_per_frame

def inference_gnd_video(predictor, scene):
    frame_dir = f"{scene['data_path']}/images"
    gnd_prompt = scene['gnd_prompt']
    prompts = gnd_prompt if isinstance(gnd_prompt, (list, tuple)) else [gnd_prompt]

    # Start
    start_response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": frame_dir,
        }
    )
    session_id = start_response["session_id"]

    combined_mask_per_frame = {}
    for prompt in prompts:
        # accept either a single prompt string or an iterable of prompt strings
        response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

        # we will just propagate from frame 0 to the end of the video
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_idx = response["frame_index"]
            masks = response["outputs"].get("out_binary_masks")
            # Merge outputs from this prompt with existing frame outputs
            if len(masks) > 0:
                mask_stack = np.stack(masks, axis=0)
                combined_mask = np.any(mask_stack > 0, axis=0)

                prev_mask = combined_mask_per_frame.get(frame_idx)
                if prev_mask is not None:
                    combined_mask_per_frame[frame_idx] = combined_mask | prev_mask
                else:
                    combined_mask_per_frame[frame_idx] = combined_mask
            else:
                combined_mask_per_frame[frame_idx] = None
        
        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
    
    # finally, close the inference session to free its GPU resources
    # (you may start a new session on another video)
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    return combined_mask_per_frame

def sam_inference_a_scene(scene):
    logger.info(f"SAM Inference on scene: {scene['exp_name']}")
    torch.inference_mode().__enter__()

    # predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
    predictor = build_sam3_video_predictor()

    # Building masks
    combined_mask_per_frame = inference_bldg_video(predictor, scene)
    saved, no_mask, out_dir = save_masks_by_frame_index(
        combined_mask_per_frame,
        f"{scene['data_path']}/images",
        scene['data_path'],
        'bldg_masks',
    )
    logger.info(f"{scene['exp_name']}: saved {saved} bldg_masks, {no_mask} of them are empty")

    # Ground masks
    combined_mask_per_frame = inference_gnd_video(predictor, scene)
    saved, no_mask, out_dir = save_masks_by_frame_index(
        combined_mask_per_frame,
        f"{scene['data_path']}/images",
        scene['data_path'],
        'gnd_masks',
    )
    logger.info(f"{scene['exp_name']}: saved {saved} gnd_masks, {no_mask} of them are empty")

    # after all inference is done, we can shutdown the predictor
    # to free up the multi-GPU process group
    predictor.shutdown()

def sam_inference_all_scenes(scenes):
    torch.inference_mode().__enter__()
    # predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
    predictor = build_sam3_video_predictor()

    for scene in scenes:
        logger.info(f"SAM Inference on scene: {scene['exp_name']}")

        # Building masks
        combined_mask_per_frame = inference_bldg_video(predictor, scene)
        saved, no_mask, out_dir = save_masks_by_frame_index(
            combined_mask_per_frame,
            f"{scene['data_path']}/images",
            scene['data_path'],
            'bldg_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} bldg_masks, {no_mask} of them are empty")

        # Ground masks
        combined_mask_per_frame = inference_gnd_video(predictor, scene)
        saved, no_mask, out_dir = save_masks_by_frame_index(
            combined_mask_per_frame,
            f"{scene['data_path']}/images",
            scene['data_path'],
            'gnd_masks',
        )
        logger.info(f"{scene['exp_name']}: saved {saved} gnd_masks, {no_mask} of them are empty")

    # after all inference is done, we can shutdown the predictor
    # to free up the multi-GPU process group
    predictor.shutdown()
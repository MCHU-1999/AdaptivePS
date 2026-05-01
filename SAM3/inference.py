import argparse
import os
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

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

def run_sequence_demo(resource_path: str, prompt: str):
    """
    Run sequence/video SAM3 inference.

    resource_path can be:
    - a folder of JPEG frames
    - an MP4 file
    """
    predictor = build_sam3_video_predictor()

    start_response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": resource_path,
        }
    )
    session_id = start_response["session_id"]

    _ = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
    )

    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"[sequence] path={resource_path} prompt='{prompt}' -> {len(outputs_per_frame)} frames")
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 inference for a single image or a sequence folder/video.")
    parser.add_argument("path", type=str, help="Path to a single image, image folder, or video file")
    parser.add_argument("--prompt", required=True, help="Text prompt to segment")
    args = parser.parse_args()
    path = args.path

    if os.path.isdir(path):
        run_sequence_demo(path, args.prompt)
        exit(0)

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv"}:
            run_sequence_demo(path, args.prompt)
        else:
            raise NotImplementedError("On no!")
        exit(0)

    raise ValueError(f"Path does not exist: {path}")
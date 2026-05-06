import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from loguru import logger

from sam3 import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.eval.postprocessors import PostProcessImage

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_GLOBAL_QUERY_COUNTER = 1


def set_hf_token_from_txt(filepath: str = "./hf_token.txt"):
    with open(filepath, "r") as f:
        token = f.read().strip()

    os.environ["HF_TOKEN"] = token
    logger.info("HF_TOKEN set.")


def list_sorted_frames(data_dir: str) -> List[str]:
    frame_files = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full_path) and ext in IMAGE_EXTS:
            frame_files.append(name)

    frame_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return frame_files


def _create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def _set_image(datapoint: Datapoint, pil_image: Image.Image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def _add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    global _GLOBAL_QUERY_COUNTER

    assert len(datapoint.images) == 1, "Please set image before adding prompt"

    # SAMImage.size is [h, w]
    img_size = datapoint.images[0].size
    img_h, img_w = img_size[0], img_size[1]
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_QUERY_COUNTER,
                original_image_id=_GLOBAL_QUERY_COUNTER,
                original_category_id=1,
                original_size=[img_h, img_w],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    _GLOBAL_QUERY_COUNTER += 1
    return _GLOBAL_QUERY_COUNTER - 1

def _build_transform():
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _build_postprocessor():
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )


def _ensure_numpy_mask(mask) -> np.ndarray:
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    mask_np = np.asarray(mask, dtype=np.float32)
    mask_np = np.squeeze(mask_np)
    return (mask_np > 0.5).astype(bool)


def _empty_mask_for_image(img_shape: Tuple[int, int]) -> np.ndarray:
    h, w = img_shape
    return np.zeros((h, w), dtype=bool)


def _largest_instance_mask(result: Dict, fallback_shape: Tuple[int, int]) -> np.ndarray:
    masks = result.get("masks", None)
    boxes = result.get("boxes", None)

    if masks is None or masks is False:
        return np.zeros(fallback_shape, dtype=bool)

    try:
        n_inst = int(len(masks))
    except (TypeError, AttributeError):
        return np.zeros(fallback_shape, dtype=bool)
    
    if n_inst == 0:
        return np.zeros(fallback_shape, dtype=bool)

    best_idx = 0

    if boxes is not None and len(boxes) == n_inst:
        # boxes are tensors, convert to float to compute areas
        boxes_t = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
        x1 = boxes_t[:, 0]
        y1 = boxes_t[:, 1]
        x2 = boxes_t[:, 2]
        y2 = boxes_t[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        best_idx = int(torch.argmax(areas).item())
    else:
        # Fallback: pick largest mask area when bbox is unavailable.
        mask_areas = []
        for i in range(n_inst):
            m = _ensure_numpy_mask(masks[i])
            mask_areas.append(int(m.sum()))
        best_idx = int(np.argmax(mask_areas))

    return _ensure_numpy_mask(masks[best_idx])


def _or_all_instances_mask(result: Dict, fallback_shape: Tuple[int, int]) -> np.ndarray:
    masks = result.get("masks", None)
    if masks is None or masks is False:
        return np.zeros(fallback_shape, dtype=bool)
    
    try:
        if len(masks) == 0:
            return np.zeros(fallback_shape, dtype=bool)
    except TypeError:
        return np.zeros(fallback_shape, dtype=bool)

    agg = None
    for i in range(len(masks)):
        cur = _ensure_numpy_mask(masks[i])
        if cur.shape != fallback_shape:
            cur = np.zeros(fallback_shape, dtype=bool)
        agg = cur if agg is None else (agg | cur)

    if agg is None:
        agg = np.zeros(fallback_shape, dtype=bool)
    return agg


def _save_masks_by_filename(
    mask_by_frame: Dict[str, np.ndarray],
    frame_dir: str,
    output_root_dir: str,
    mask_dir_name: str,
    img_shape: Tuple[int, int],
):
    frame_files = list_sorted_frames(frame_dir)
    out_dir = os.path.join(output_root_dir, mask_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    no_mask = 0

    for frame_name in frame_files:
        mask = mask_by_frame.get(frame_name)
        if mask is None:
            mask = _empty_mask_for_image(img_shape)
            no_mask += 1
        else:
            # Ensure mask is boolean/binary
            mask = mask.astype(bool) if not mask.dtype == bool else mask
            if int(mask.sum()) == 0:
                no_mask += 1

        # Convert to uint8 for saving (0-255)
        mask_uint8 = mask.astype(np.uint8) * 255
        out_path = os.path.join(out_dir, frame_name)
        Image.fromarray(mask_uint8, mode="L").save(out_path)
        saved += 1

    return saved, no_mask, out_dir


def _get_image_shape(frame_dir: str) -> Tuple[int, int]:
    frame_files = list_sorted_frames(frame_dir)
    if not frame_files:
        raise ValueError(f"No image files found in {frame_dir}")

    sample_path = os.path.join(frame_dir, frame_files[0])
    with Image.open(sample_path) as img:
        return img.height, img.width


def _run_batched_image_inference_for_prompts(
    scene: Dict,
    model,
    transform,
    postprocessor,
    prompts: List[str],
    batch_size: int,
):
    frame_dir = f"{scene['data_path']}/images"
    frame_files = list_sorted_frames(frame_dir)

    results_per_prompt_per_frame: Dict[str, Dict[str, Dict]] = {p: {} for p in prompts}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for start in range(0, len(frame_files), batch_size):
        chunk_files = frame_files[start : start + batch_size]

        datapoints = []
        query_to_meta = {}

        for frame_name in chunk_files:
            frame_path = os.path.join(frame_dir, frame_name)
            img = Image.open(frame_path).convert("RGB")

            dp = _create_empty_datapoint()
            _set_image(dp, img)

            for prompt in prompts:
                qid = _add_text_prompt(dp, prompt)
                query_to_meta[qid] = (frame_name, prompt, (img.height, img.width))

            dp = transform(dp)
            datapoints.append(dp)

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)

        output = model(batch)
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        for qid, (frame_name, prompt, _shape) in query_to_meta.items():
            if qid in processed_results:
                results_per_prompt_per_frame[prompt][frame_name] = processed_results[qid]
            else:
                results_per_prompt_per_frame[prompt][frame_name] = {}

    return results_per_prompt_per_frame


def inference_bldg_mask(scene: Dict, model, transform, postprocessor, batch_size: int = 4):
    bldg_prompt = scene["bldg_prompt"]
    frame_dir = f"{scene['data_path']}/images"
    frame_files = list_sorted_frames(frame_dir)
    img_shape = _get_image_shape(frame_dir)

    prompt_results = _run_batched_image_inference_for_prompts(
        scene=scene,
        model=model,
        transform=transform,
        postprocessor=postprocessor,
        prompts=[bldg_prompt],
        batch_size=batch_size,
    )

    per_frame_result = prompt_results[bldg_prompt]
    bldg_masks: Dict[str, np.ndarray] = {}

    for frame_name in frame_files:
        result = per_frame_result.get(frame_name, {})
        bldg_masks[frame_name] = _largest_instance_mask(result, fallback_shape=img_shape)

    return bldg_masks


def inference_gnd_mask(scene: Dict, model, transform, postprocessor, batch_size: int = 4):
    gnd_prompt = scene["gnd_prompt"]
    prompts = gnd_prompt if isinstance(gnd_prompt, (list, tuple)) else [gnd_prompt]

    frame_dir = f"{scene['data_path']}/images"
    frame_files = list_sorted_frames(frame_dir)
    img_shape = _get_image_shape(frame_dir)

    prompt_results = _run_batched_image_inference_for_prompts(
        scene=scene,
        model=model,
        transform=transform,
        postprocessor=postprocessor,
        prompts=list(prompts),
        batch_size=batch_size,
    )

    gnd_masks: Dict[str, np.ndarray] = {}

    for frame_name in frame_files:
        merged = np.zeros(img_shape, dtype=bool)
        for prompt in prompts:
            result = prompt_results[prompt].get(frame_name, {})
            merged = merged | _or_all_instances_mask(result, fallback_shape=img_shape)

        gnd_masks[frame_name] = merged

    return gnd_masks


def sam_inference_a_scene(scene: Dict, model, transform, postprocessor, batch_size: int = 4):
    logger.info(f"SAM Inference on scene: {scene['exp_name']}")
    frame_dir = f"{scene['data_path']}/images"
    img_shape = _get_image_shape(frame_dir)

    bldg_masks = inference_bldg_mask(scene, model, transform, postprocessor, batch_size=batch_size)
    saved, no_mask, _out_dir = _save_masks_by_filename(
        bldg_masks,
        frame_dir,
        scene["data_path"],
        "bldg_masks",
        img_shape,
    )
    logger.info(f"{scene['exp_name']}: saved {saved} bldg_masks, {no_mask} are empty")

    gnd_masks = inference_gnd_mask(scene, model, transform, postprocessor, batch_size=batch_size)
    saved, no_mask, _out_dir = _save_masks_by_filename(
        gnd_masks,
        frame_dir,
        scene["data_path"],
        "gnd_masks",
        img_shape,
    )
    logger.info(f"{scene['exp_name']}: saved {saved} gnd_masks, {no_mask} are empty")


def sam_inference_all_scenes(scenes: List[Dict], batch_size: int = 50):
    # Global performance setup.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    torch.inference_mode().__enter__()

    model = build_sam3_image_model()
    transform = _build_transform()
    postprocessor = _build_postprocessor()

    for scene in scenes:
        sam_inference_a_scene(
            scene,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            batch_size=batch_size,
        )

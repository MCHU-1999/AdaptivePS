import run_vanilla
import os
from DA3.inference_dtu import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_adaptivePS
from run_vanilla import run_vanilla


# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
BD_NUM = [
    0, 6, 9, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 28, 29, 43, 44, 46
]
BD_SCENES = [
    # DTU Building Datasets
    {
        "exp_name": f"scan{scan_num}",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building/scan{scan_num}",
        "bldg_prompt": "houses/buildings",
        "bldg_mask_mode": "squash"
    } for scan_num in BD_NUM
]
BD_LIKE_SCENES = [
    # DTU Building-like Datasets
    {
        "exp_name": "scan10",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan10",
        "bldg_prompt": "box",
        "bldg_mask_mode": "squash"
    },
    {
        "exp_name": "scan13",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan13",
        "bldg_prompt": "box",
        "bldg_mask_mode": "squash"
    },
    {
        "exp_name": "scan34",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan34",
        "bldg_prompt": "bricks",
        "bldg_mask_mode": "squash"
    },
    {
        "exp_name": "scan40",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan40",
        "bldg_prompt": "bricks",
        "bldg_mask_mode": "squash"
    },
    {
        "exp_name": "scan47",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan47",
        "bldg_prompt": "wave signal generator",
        "bldg_mask_mode": "squash"
    },
    {
        "exp_name": "scan77",
        "data_path": f"{MY_STORAGE}/DTU_ALL/Building_like/scan77",
        "bldg_prompt": "Coffee Mokkapot",
        "bldg_mask_mode": "squash"
    },
]


if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    AllDTU = BD_SCENES + BD_LIKE_SCENES

    ## SAM3
    # sam_inference_all_scenes(AllDTU)

    ## DA3
    # da3_inference_all_scenes(AllDTU)

    ## AdaptivePS
    for scene in BD_SCENES:
        run_adaptivePS(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="AdaptivePS/DTU-Building",
            conf_path="configs/DA3FG++DTU.conf",
            mask="bldg_masks"
        )
    for scene in BD_LIKE_SCENES:
        run_adaptivePS(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="AdaptivePS/DTU-Building-like",
            conf_path="configs/DA3FG++DTU.conf",
            mask="bldg_masks"
        )

    ## Vanilla PlanarSplatting
    for scene in BD_SCENES:
        run_vanilla(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="Vanilla/DTU-Building",
            conf_path="configs/vanilla-DTU.conf",
        )
    for scene in BD_LIKE_SCENES:
        run_vanilla(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="Vanilla/DTU-Building-like",
            conf_path="configs/vanilla-DTU.conf",
        )
import os
from DA3.inference_dtu import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_APS import run_adaptivePS
from run_baseline import run_baseline


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
        "bldg_mask_mode": "squash",
    } for scan_num in BD_NUM
]


if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    ## SAM3
    sam_inference_all_scenes(BD_SCENES)

    ## DA3
    # da3_inference_all_scenes(BD_SCENES)

    ## AdaptivePS
    for scene in BD_SCENES:
        run_adaptivePS(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="AdaptivePS/DTU-Building",
            conf_path="configs/APS-DTU.conf",
            mask="bldg_masks"
        )

    ## Baseline PlanarSplatting
    for scene in BD_SCENES:
        run_baseline(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="Baseline/DTU-Building",
            conf_path="configs/baseline-DTU.conf",
        )
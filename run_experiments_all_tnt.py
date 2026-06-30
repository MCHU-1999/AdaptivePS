import subprocess, sys, os
# from DA3.inference_colmap import da3_inference_all_scenes, da3_inference_a_scene
from DA3.inference import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_APS import run_adaptivePS
from run_baseline import run_baseline


# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
SCENES = [
    # TnT Datasets
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn_sparse20",
        "bldg_prompt": "The building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"],
        "depth_trunc": 6,
        "max_depth": 200,
    },
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn_sparse40",
        "bldg_prompt": "The building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"],
        "depth_trunc": 6,
        "max_depth": 200,
    },
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn_sparse80",
        "bldg_prompt": "The building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"],
        "depth_trunc": 6,
        "max_depth": 200,
    },
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
        "bldg_prompt": "The building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"],
        "depth_trunc": 6,
        "max_depth": 200,
    },
]


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    # ## SAM3
    # sam_inference_all_scenes(SCENES)

    # ## DA3
    # da3_inference_all_scenes(SCENES)

    ## PlanarSplatting
    scene = SCENES[0]
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS-sparse/20_2",
    #     conf_path="configs/APS-big2.conf",
    #     mask="bldg_masks"
    # )
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="AdaptivePS-sparse/20_3",
        conf_path="configs/APS-big3.conf",
        mask="bldg_masks"
    )

    scene = SCENES[-1]
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS/Barn_2",
    #     conf_path="configs/APS-Barn2.conf",
    #     mask="bldg_masks"
    # )
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="AdaptivePS/Barn_3",
        conf_path="configs/APS-Barn3.conf",
        mask="bldg_masks"
    )

    # scene = SCENES[1]
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS-sparse/40_2",
    #     conf_path="configs/APS-big2.conf",
    #     mask="bldg_masks"
    # )
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS-sparse/40_3",
    #     conf_path="configs/APS-big3.conf",
    #     mask="bldg_masks"
    # )

    # scene = SCENES[2]
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS-sparse/80_2",
    #     conf_path="configs/APS-big2.conf",
    #     mask="bldg_masks"
    # )
    # run_adaptivePS(
    #     data_path=scene['data_path'],
    #     exp_name=scene['exp_name'],
    #     out_path="AdaptivePS-sparse/80_3",
    #     conf_path="configs/APS-big3.conf",
    #     mask="bldg_masks"
    # )
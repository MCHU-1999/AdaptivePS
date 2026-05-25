import os
from DA3.inference_dtu import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_adaptivePS
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
        "bldg_mask_mode": "squash"
    } for scan_num in BD_NUM
]


if __name__ == "__main__":

    # ## Swap normal source
    # for scene in BD_SCENES:
    #     run_adaptivePS(
    #         data_path=scene['data_path'],
    #         exp_name=scene['exp_name'],
    #         out_path="Ablation/Normalswap",
    #         conf_path="configs/DA3FG++DTU.conf",
    #         mask="bldg_masks",
    #         runtime_log_path="evaluation/runtime_logs/ablation_normalswap.json"
    #     )

    ## None of the 3
    for scene in BD_SCENES:
        run_adaptivePS(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="Ablation/Allnone",
            conf_path="configs/DA3FG++DTU-none.conf",
            mask="bldg_masks",
            runtime_log_path="evaluation/runtime_logs/ablation_allnone.json"
        )

    # ## only Mesh post-processing
    # for scene in BD_SCENES:
    #     run_adaptivePS(
    #         data_path=scene['data_path'],
    #         exp_name=scene['exp_name'],
    #         out_path="Ablation/Only1mesh",
    #         conf_path="configs/DA3FG++DTU-only1mesh.conf",
    #         mask="bldg_masks",
    #         runtime_log_path="evaluation/runtime_logs/ablation_only1mesh.json"
    #     )

    # ## only Mask-Guided Densification & Pruning
    # for scene in BD_SCENES:
    #     run_adaptivePS(
    #         data_path=scene['data_path'],
    #         exp_name=scene['exp_name'],
    #         out_path="Ablation/Onlysplit",
    #         conf_path="configs/DA3FG++DTU-onlysplit.conf",
    #         mask="bldg_masks",
    #         runtime_log_path="evaluation/runtime_logs/ablation_onlysplit.json"
    #     )

    # ## only Final Mask-Guided Trim
    # for scene in BD_SCENES:
    #     run_adaptivePS(
    #         data_path=scene['data_path'],
    #         exp_name=scene['exp_name'],
    #         out_path="Ablation/Onlytrim",
    #         conf_path="configs/DA3FG++DTU-onlytrim.conf",
    #         mask="bldg_masks",
    #         runtime_log_path="evaluation/runtime_logs/ablation_onlytrim.json"
    #     )



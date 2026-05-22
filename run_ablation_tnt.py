import subprocess, sys, os
from DA3.inference_colmap import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_adaptivePS
from run_vanilla import run_vanilla


# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
# TnT Dataset
scene = {
    "exp_name": "Barn",
    "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
    "bldg_prompt": "The barn house in foreground",
    "bldg_mask_mode": "biggest",
    "gnd_prompt": ["ground", "grass", "pavement"],
    "depth_trunc": 6,
    "max_depth": 200,
}


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":


    ## w/o Mesh post-processing
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/No1mesh",
        conf_path="configs/DA3FG++Barn-no1mesh.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_no1mesh.json"
    )

    ## w/o Mask-Guided Densification & Pruning
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Nosplit",
        conf_path="configs/DA3FG++Barn-nosplit.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_nosplit.json"
    )

    ## w/o Final Mask-Guided Trim
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Notrim",
        conf_path="configs/DA3FG++Barn-notrim.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_notrim.json"
    )

    ## only Mesh post-processing
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Only1mesh",
        conf_path="configs/DA3FG++Barn-only1mesh.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_only1mesh.json"
    )

    ## only Mask-Guided Densification & Pruning
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Onlysplit",
        conf_path="configs/DA3FG++Barn-onlysplit.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_onlysplit.json"
    )

    ## only Final Mask-Guided Trim
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Onlytrim",
        conf_path="configs/DA3FG++Barn-onlytrim.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_onlytrim.json"
    )

    ## None of the 3
    run_adaptivePS(
        data_path=scene['data_path'],
        exp_name=scene['exp_name'],
        out_path="Ablation_tnt/Allnone",
        conf_path="configs/DA3FG++Barn-none.conf",
        mask="bldg_masks",
        runtime_log_path="evaluation/runtime_logs_tnt/ablation_allnone.json"
    )
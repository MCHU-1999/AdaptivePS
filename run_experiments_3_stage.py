import subprocess, sys, os
from DA3.inference import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_planarsplatting

from run_experiments import SCENES


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    ## SAM3
    # sam_inference_all_scenes(SCENES)

    ## DA3
    # da3_inference_all_scenes(SCENES)

    ## PlanarSplatting
    for scene in SCENES:
        run_planarsplatting(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="A3_progress/DA3FG2_split",
            conf_path="configs/DA3FG++big.conf",
            mask="bldg_masks"
        )
import subprocess, sys, os
from DA3.inference import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_planarsplatting


# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
SCENES = [
    # # TnT Datasets
    # {
    #     "exp_name": "Barn",
    #     "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
    #     "bldg_prompt": "The barn house in foreground",
    #     "bldg_mask_mode": "biggest",
    #     "gnd_prompt": ["ground", "grass", "pavement"]
    # },
    # # DTU Datasets
    # {
    #     "exp_name": "dtu-scan24",
    #     "data_path": f"{MY_STORAGE}/DTU/scan24",
    #     "bldg_prompt": "the buildings in foreground",
    #     "bldg_mask_mode": "squash",
    #     "gnd_prompt": "white table surface"
    # },
    # {
    #     "exp_name": "dtu-scan40",
    #     "data_path": f"{MY_STORAGE}/DTU/scan40",
    #     "bldg_prompt": "the bricks",
    #     "bldg_mask_mode": "squash",
    #     "gnd_prompt": "white table surface"
    # },
    # Pexels Datasets
    {
        "exp_name": "church-cadeby",
        "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
        "bldg_prompt": "that stone masonry church building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass"]
    },
    {
        "exp_name": "church-chesterfield",
        "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
        "bldg_prompt": "the red building with a spire in center of frame",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["grass", "road", "pavement"]
    },
    {
        "exp_name": "killingbeck-cemetery",
        "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
        "bldg_prompt": "that stone masonry church building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "road"]
    },
    {
        "exp_name": "moskee-haarlem",
        "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
        "bldg_prompt": "that building in the center of frame",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["water", "grass"]
    },
    {
        "exp_name": "tower-court",
        "data_path": f"{MY_STORAGE}/Pexels/tower-court",
        "bldg_prompt": "that building with clock-tower",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "road", "pavement"]
    },
    {
        "exp_name": "wotrubakirche",
        "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
        "bldg_prompt": "the modernism concrete building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["grass", "road", "pavement"]
    },
    {
        "exp_name": "elbphilharmonie",
        "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
        "bldg_prompt": "Elbphilharmonie, that modernism red-brick and glass building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["water", "road", "pavement"]
    },
    {
        "exp_name": "krasna-horka-castle",
        "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
        "bldg_prompt": "that castle building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"]
    },
    {
        "exp_name": "clocktower",
        "data_path": f"{MY_STORAGE}/Pexels/clocktower",
        "bldg_prompt": "that clocktower",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "pavement"]
    }
]


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    ## SAM3
    # sam_inference_all_scenes(SCENES)
    sam_inference_a_scene(SCENES[-1])

    ## DA3
    da3_inference_all_scenes(SCENES)
    # da3_inference_a_scene(SCENES[-1])

    ## PlanarSplatting
    for scene in SCENES:
        run_planarsplatting(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="A3_progress/Pexels",
            conf_path="configs/DA3FG++big.conf",
            mask="bldg_masks"
        )
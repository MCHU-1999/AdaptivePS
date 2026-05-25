import subprocess, sys, os
from DA3.inference import da3_inference_all_scenes, da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes, sam_inference_a_scene
from run_DA3FG import run_adaptivePS
from run_baseline import run_baseline

# =================================================================
# SUMMARY OF DEPTHS PER SCENE (took 50 samples only)
# =================================================================
# Scene                     | 95th Percentile | Absolute Max
# -----------------------------------------------------------------
# church-cadeby             | 21.5026         | 67.3038
# church-chesterfield       | 21.8328         | 91.7950
# killingbeck-cemetery      | 20.7911         | 34.5902
# moskee-haarlem            | 32.5062         | 50.0993
# tower-court               | 15.8487         | 61.1334
# wotrubakirche             | 26.1040         | 53.9619
# elbphilharmonie           | 40.5135         | 140.6320
# krasna-horka-castle       | 45.9030         | 115.7443
# clocktower                | 16.0855         | 82.8168
# -----------------------------------------------------------------
# GLOBAL MAXIMUM (of the 95th percentiles): 45.9030
# =================================================================

# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
SCENES = [
    # Pexels Datasets
    {
        "exp_name": "church-cadeby",
        "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
        "bldg_prompt": "that stone masonry church building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass"],
        "depth_trunc": 25,
        "max_depth": 200,
    },
    {
        "exp_name": "church-chesterfield",
        "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
        "bldg_prompt": "the red building with a spire in center of frame",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["grass", "road", "pavement"],
        "depth_trunc": 25,
        "max_depth": 200,
    },
    {
        "exp_name": "killingbeck-cemetery",
        "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
        "bldg_prompt": "that stone masonry church building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "road"],
        "depth_trunc": 25,
        "max_depth": 200,
    },
    {
        "exp_name": "moskee-haarlem",
        "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
        "bldg_prompt": "that building in the center of frame",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["water", "grass"],
        "depth_trunc": 35,
        "max_depth": 200,
    },
    {
        "exp_name": "tower-court",
        "data_path": f"{MY_STORAGE}/Pexels/tower-court",
        "bldg_prompt": "that building with clock-tower",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "road", "pavement"],
        "depth_trunc": 20,
        "max_depth": 200,
    },
    {
        "exp_name": "wotrubakirche",
        "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
        "bldg_prompt": "the modernism concrete building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["grass", "road", "pavement"],
        "depth_trunc": 30,
        "max_depth": 200,
    },
    {
        "exp_name": "elbphilharmonie",
        "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
        "bldg_prompt": "Elbphilharmonie, that modernism red-brick and glass building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["water", "road", "pavement"],
        "depth_trunc": 45,
        "max_depth": 200,
    },
    {
        "exp_name": "krasna-horka-castle",
        "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
        "bldg_prompt": "that castle building",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "grass", "pavement"],
        "depth_trunc": 48,
        "max_depth": 200,
    },
    {
        "exp_name": "clocktower",
        "data_path": f"{MY_STORAGE}/Pexels/clocktower",
        "bldg_prompt": "that clocktower",
        "bldg_mask_mode": "biggest",
        "gnd_prompt": ["ground", "pavement"],
        "depth_trunc": 18,
        "max_depth": 200,
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
    sam_inference_all_scenes(SCENES)

    ## DA3
    da3_inference_all_scenes(SCENES)

    ## PlanarSplatting
    for scene in SCENES:
        run_adaptivePS(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="AdaptivePS/Pexels",
            conf_path="configs/DA3FG++big.conf",
            mask="bldg_masks"
        )
    for scene in SCENES:
        run_baseline(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="Vanilla/Pexels",
            conf_path="configs/vanilla-big.conf",
            depth_trunc=scene['depth_trunc'],
        )
            

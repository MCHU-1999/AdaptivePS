import os
from SAM3.inference import set_hf_token_from_txt, inference_bd_gnd


# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
SCENES = [
    # TnT Datasets
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
        "bldg_prompt": "house in front",
        "gnd_prompt": ["ground", "grass", "pavement"]
    },
    # DTU Datasets
    {
        "exp_name": "dtu-scan24",
        "data_path": f"{MY_STORAGE}/DTU/scan24",
        "bldg_prompt": "the buildings in foreground",
        "gnd_prompt": "white table surface"
    },
    {
        "exp_name": "dtu-scan40",
        "data_path": f"{MY_STORAGE}/DTU/scan40",
        "bldg_prompt": "the bricks",
        "gnd_prompt": "white table surface"
    },
    # Pexels Datssets
    {
        "exp_name": "church-cadeby",
        "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
        "bldg_prompt": "that stone masonry church building",
        "gnd_prompt": ["ground", "grass", "pavement"]
    },
    {
        "exp_name": "church-chesterfield",
        "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
        "bldg_prompt": "that modern black-roofed red-brick church building with a spire",
        "gnd_prompt": ["ground", "grass", "road", "pavement"]
    },
    {
        "exp_name": "killingbeck-cemetery",
        "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
        "bldg_prompt": "that stone masonry church building",
        "gnd_prompt": ["ground", "grass", "road", "pavement"]
    },
    {
        "exp_name": "moskee-haarlem",
        "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
        "bldg_prompt": "that building in the center of frame",
        "gnd_prompt": ["ground", "water", "grass", "road", "pavement"]
    },
    {
        "exp_name": "tower-court",
        "data_path": f"{MY_STORAGE}/Pexels/tower-court",
        "bldg_prompt": "that historic red-brick clock-tower building",
        "gnd_prompt": ["ground", "grass", "road", "pavement"]
    },
    {
        "exp_name": "wotrubakirche",
        "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
        "bldg_prompt": "the modernism concrete building",
        "gnd_prompt": ["ground", "grass", "road", "pavement"]
    },
    {
        "exp_name": "elbphilharmonie",
        "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
        "bldg_prompt": "Elbphilharmonie, that modernism red-brick and glass building",
        "gnd_prompt": ["ground", "water", "road", "pavement"]
    },
    {
        "exp_name": "krasna-horka-castle",
        "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
        "bldg_prompt": "that castle building",
        "gnd_prompt": ["ground", "grass", "pavement"]
    }
]


# Main Function
if __name__ == "__main__":

    if not os.environ.get("HF_TOKEN"):
        token_path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
        if os.path.exists(token_path):
            set_hf_token_from_txt(token_path)
        else:
            raise FileNotFoundError(
                f"HF_TOKEN is not set and token file was not found: {token_path}"
            )

    inference_bd_gnd(SCENES)
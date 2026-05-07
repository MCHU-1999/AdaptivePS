import os
from SAM3.inference import set_hf_token_from_txt, sam_inference_all_scenes

# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
BD_SCAN_NUM = [
    6, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 43, 44, 46
]
SCENES = [
    # DTU Full Datasets
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/DTU_full/Building/scan{scan_num}",
        "bldg_prompt": "houses/buildings",
        "gnd_prompt": "white table surface",
        "bldg_mask_mode": "squash"
    } for scan_num in BD_SCAN_NUM
]


if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        token_path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
        if os.path.exists(token_path):
            set_hf_token_from_txt(token_path)
        else:
            raise FileNotFoundError(
                f"HF_TOKEN is not set and token file was not found: {token_path}"
            )

    sam_inference_all_scenes(SCENES)
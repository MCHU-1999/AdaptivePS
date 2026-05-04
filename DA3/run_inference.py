import os
import subprocess

# CONST
MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
SCENES = [
    # TnT Datasets
    {
        "exp_name": "Barn",
        "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
    },
    # DTU Datasets
    {
        "exp_name": "dtu-scan24",
        "data_path": f"{MY_STORAGE}/DTU/scan24",
    },
    {
        "exp_name": "dtu-scan40",
        "data_path": f"{MY_STORAGE}/DTU/scan40",
    },
    # Pexels Datssets
    {
        "exp_name": "church-cadeby",
        "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
    },
    {
        "exp_name": "church-chesterfield",
        "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
    },
    {
        "exp_name": "killingbeck-cemetery",
        "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
    },
    {
        "exp_name": "moskee-haarlem",
        "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
    },
    {
        "exp_name": "tower-court",
        "data_path": f"{MY_STORAGE}/Pexels/tower-court",
    },
    {
        "exp_name": "wotrubakirche",
        "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
    },
    {
        "exp_name": "elbphilharmonie",
        "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
    },
    {
        "exp_name": "krasna-horka-castle",
        "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
    }
]


# Main Function
if __name__ == "__main__":

    for scene in SCENES:
        exp_name = scene["exp_name"]
        data_path = scene["data_path"]
        
        print(f"Processing: {exp_name}")
        print(f"Data path: {data_path}")
        
        cmd = [
            "python", "DA3/inference_w_masks.py",
            "--data_dir", data_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Completed: {exp_name}\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {exp_name} (exit code: {e.returncode})\n")
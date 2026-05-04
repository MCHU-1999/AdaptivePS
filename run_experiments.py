import subprocess
import sys
import os

def run_script(script_name: str, scenes: list[dict], extra_args: None|list[str] = None):
    print(f"Total scenes to run: {len(scenes)}")
    if not os.path.exists(script_name):
        print(f"Error: Could not find {script_name}")
        return

    for i, scene in enumerate(scenes):
        print(f"Running script: {script_name}\nExperiment: {scene['exp_name']}")

        # Construct the command
        cmd = [sys.executable, script_name]
        for key, value in scene.items():
            if value is None:
                continue
            cmd.extend([f"--{key}", str(value)])
        if extra_args:
            cmd += extra_args

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Experiment {scene['exp_name']} failed with error code {e.returncode} !!!")

def run_vanilla(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'init_mesh' in scene:
            scene.pop('init_mesh')
        if 'geo_data_path' in scene:
            scene.pop('geo_data_path')
        if 'mask' in scene:
            scene.pop('mask')
    
    run_script("run_vanilla.py", scenes_copy, extra_args)

def run_DA3(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'init_mesh' in scene:
            scene.pop('init_mesh')
        if 'geo_data_path' in scene:
            scene.pop('geo_data_path')
        if 'mask' in scene:
            scene.pop('mask')

    run_script("run_DA3.py", scenes_copy, extra_args)

def run_DA3FG(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'init_mesh' in scene:
            scene.pop('init_mesh')
        if 'geo_data_path' in scene:
            scene.pop('geo_data_path')

    run_script("run_DA3FG.py", scenes_copy, extra_args)


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # CONST
    MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
    SCENES = [
        # TnT Datasets
        {
            "exp_name": "Barn",
            "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
            "mask": "bldg_masks"
        },
        # DTU Datasets
        {
            "exp_name": "dtu-scan24",
            "data_path": f"{MY_STORAGE}/DTU/scan24",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "dtu-scan40",
            "data_path": f"{MY_STORAGE}/DTU/scan40",
            "mask": "bldg_masks"
        },
        # Pexel Datasets
        {
            "exp_name": "church-cadeby",
            "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "church-chesterfield",
            "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "killingbeck-cemetery",
            "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "moskee-haarlem",
            "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "tower-court",
            "data_path": f"{MY_STORAGE}/Pexels/tower-court",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "wotrubakirche",
            "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "elbphilharmonie",
            "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
            "mask": "bldg_masks"
        },
        {
            "exp_name": "krasna-horka-castle",
            "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
            "mask": "bldg_masks"
        }
    ]

    # Starts here
    # Capture any extra arguments passed to this script (like --data_path) to forward them
    extra_args = sys.argv[1:]

    ## ==================== DA3FG
    this_extra = [
        '--out_path', 'A3_progress/DA3FG2_split',
        '--conf_path', 'configs/DA3FG++big.conf'
    ]
    run_DA3FG(SCENES, this_extra)

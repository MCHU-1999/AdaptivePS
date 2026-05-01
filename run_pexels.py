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
        {
            "exp_name": "church-cadeby",
            "data_path": f"{MY_STORAGE}/Pexels/church-cadeby",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "church-chesterfield",
            "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "clocktower",
            "data_path": f"{MY_STORAGE}/Pexels/clocktower",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "killingbeck-cemetery",
            "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "moskee-haarlem",
            "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "tower-court",
            "data_path": f"{MY_STORAGE}/Pexels/tower-court",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "wotrubakirche",
            "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        },
        {
            "exp_name": "yorkshire-post",
            "data_path": f"{MY_STORAGE}/Pexels/yorkshire-post",
            "voxel_length": 0.04,
            "max_depth": 250.0,
            "mask": "bldg_masks"
        }
    ]

    # Starts here
    # Capture any extra arguments passed to this script (like --data_path) to forward them
    extra_args = sys.argv[1:]

    # ## ==================== vanilla
    # this_extra = [
    #     '--out_path', 'A2_progress/vanilla_bgtrim',
    #     '--use_precomputed_data'
    # ]
    # run_vanilla(SCENES, this_extra)

    ## ==================== DA3
    # this_extra = [
    #     '--out_path', 'A2_progress/DA3_hardtrim',
    #     '--use_precomputed_data'
    # ]
    # run_DA3(SCENES, this_extra)

    ## ==================== DA3FG
    this_extra = [
        '--out_path', 'A2_progress/DA3FG2_split',
        '--conf_path', 'configs/DA3FG++pexels.conf'
    ]
    run_DA3FG(SCENES, this_extra)

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

def run_blender(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'init_mesh' in scene:
            scene.pop('init_mesh')
        if 'geo_data_path' in scene:
            scene.pop('geo_data_path')
        if 'mask' in scene:
            scene.pop('mask')
    
    run_script("run_blender.py", scenes_copy, extra_args)

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

def run_meshbased(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'geo_data_path' in scene:
            scene.pop('geo_data_path')
        if 'mask' in scene:
            scene.pop('mask')

    run_script("run_w_mesh.py", scenes_copy, extra_args)

def run_pgsr(scenes: list[dict], extra_args: None|list[str] = None):
    scenes_copy = [s.copy() for s in scenes]
    for scene in scenes_copy:
        if 'mask' in scene:
            scene.pop('mask')

    run_script("run_pgsr.py", scenes, extra_args)


# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # CONST
    MY_STORAGE = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
    BLENDER_SCENES = [
        {
            "exp_name": "Office_1",
            "data_path": f"{MY_STORAGE}/synthetic/Office_1",
            "voxel_length": 0.05,
            "depth_trunc": 240.0,
            "max_depth": 240.0,
        }
    ]
    SCENES = [
        {
            "exp_name": "scan24",
            "data_path": f"{MY_STORAGE}/DTU/scan24",
            "voxel_length": 0.02,
            "depth_trunc": 4.0,
            "max_depth": 20.0,
            "init_mesh": f"{MY_STORAGE}/PGSR/output_dtu/dtu_scan24/test/mesh/tsdf_fusion_post_0.ply",
            "geo_data_path": f"{MY_STORAGE}/PGSR/output_dtu/dtu_scan24/test/train/ours_30000",
            "mask": "fg_masks"
        },
        {
            "exp_name": "scan40",
            "data_path": f"{MY_STORAGE}/DTU/scan40",
            "voxel_length": 0.02,
            "depth_trunc": 4.0,
            "max_depth": 20.0,
            "init_mesh": f"{MY_STORAGE}/PGSR/output_dtu/dtu_scan40/test/mesh/tsdf_fusion_post_0.ply",
            "geo_data_path": f"{MY_STORAGE}/PGSR/output_dtu/dtu_scan40/test/train/ours_30000",
            "mask": "fg_masks"
        },
        {
            "exp_name": "Barn",
            "data_path": f"{MY_STORAGE}/TNT_GOF/TrainingSet/Barn",
            "voxel_length": 0.02,
            "depth_trunc": 6.0,
            "max_depth": 20.0,
            "init_mesh": f"{MY_STORAGE}/PGSR/output_tnt/Barn/test/mesh/tsdf_fusion_post.ply",
            "geo_data_path": f"{MY_STORAGE}/PGSR/output_tnt/Barn/test/train/ours_30000",
            "mask": "fg_masks"
        }
    ]

    # Starts here
    # Capture any extra arguments passed to this script (like --data_path) to forward them
    extra_args = sys.argv[1:]

    # ## ==================== meshbased
    # this_extra = [
    #     '--out_path', 'A2_progress/pgsrmesh_bgtrim',
    #     '--use_precomputed_data'
    # ]
    # run_meshbased(SCENES, this_extra)


    # ## ==================== vanilla
    # this_extra = [
    #     '--out_path', 'A2_progress/vanilla_bgtrim',
    #     '--use_precomputed_data'
    # ]
    # run_vanilla(SCENES, this_extra)


    # ## ==================== PGSR
    # this_extra = [
    #     '--out_path', 'A2_progress/pgsr_bgtrim',
    #     '--use_precomputed_data'
    # ]
    # run_pgsr(SCENES, this_extra)


    ## ==================== DA3
    # this_extra = [
    #     '--out_path', 'A2_progress/DA3_hardtrim',
    #     '--use_precomputed_data'
    # ]
    # run_DA3(SCENES, this_extra)

    ## ==================== DA3FG
    this_extra = [
        '--out_path', 'A2_progress/DA3FG_split',
        '--conf_path', 'configs/DA3FG++.conf',
        '--use_precomputed_data'
    ]
    run_DA3FG(SCENES, this_extra)

    this_extra = [
        '--out_path', 'A2_progress/DA3FG_moreloss',
        '--conf_path', 'configs/DA3FG+.conf',
        '--use_precomputed_data'
    ]
    run_DA3FG(SCENES, this_extra)

    ## ==================== blender
    # this_extra = [
    #     '--out_path', 'my_experiments/blender'
    # ]
    # run_blender(BLENDER_SCENES, this_extra)
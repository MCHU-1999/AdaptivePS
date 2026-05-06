import subprocess, sys, os
from DA3.inference import da3_inference_a_scene
from SAM3.inference import set_hf_token_from_txt, sam_inference_a_scene
from run_DA3FG import run_planarsplatting
from contextlib import contextmanager
from loguru import logger

@contextmanager
def stage_file_logger(scene_data_path: str, log_name: str):
    # os.makedirs(scene_data_path, exist_ok=True)
    log_file = os.path.join(scene_data_path, log_name)
    sink_id = logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} {file}:{line} {level} {message}",
        level="DEBUG"
    )
    try:
        yield
    finally:
        logger.remove(sink_id)

def run_as_subprocess(script_name: str, scene: dict, extra_args: None|list[str] = None):
    if not os.path.exists(script_name):
        print(f"Error: Could not find {script_name}")
        return

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

def run_DA3FG(scene: dict, extra_args: None|list[str] = None):
    scene_copy = scene.copy()
    if 'init_mesh' in scene_copy:
        scene_copy.pop('init_mesh')
    if 'geo_data_path' in scene_copy:
        scene_copy.pop('geo_data_path')
    if 'bldg_prompt' in scene_copy:
        scene_copy.pop('bldg_prompt')
    if 'gnd_prompt' in scene_copy:
        scene_copy.pop('gnd_prompt')

    run_as_subprocess("run_DA3FG.py", scene_copy, extra_args)


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
        "gnd_prompt": ["ground", "grass"]
    },
    {
        "exp_name": "church-chesterfield",
        "data_path": f"{MY_STORAGE}/Pexels/church-chesterfield",
        "bldg_prompt": "the red-brick black-roof building with a spire",
        "gnd_prompt": ["grass", "road", "pavement"]
    },
    {
        "exp_name": "killingbeck-cemetery",
        "data_path": f"{MY_STORAGE}/Pexels/killingbeck-cemetery",
        "bldg_prompt": "that stone masonry church building",
        "gnd_prompt": ["ground", "grass", "road"]
    },
    {
        "exp_name": "moskee-haarlem",
        "data_path": f"{MY_STORAGE}/Pexels/moskee-haarlem",
        "bldg_prompt": "that building in the center of frame",
        "gnd_prompt": ["water", "grass"]
    },
    {
        "exp_name": "tower-court",
        "data_path": f"{MY_STORAGE}/Pexels/tower-court",
        "bldg_prompt": "that building with clock-tower",
        "gnd_prompt": ["ground", "road", "pavement"]
    },
    {
        "exp_name": "wotrubakirche",
        "data_path": f"{MY_STORAGE}/Pexels/wotrubakirche",
        "bldg_prompt": "the modernism concrete building",
        "gnd_prompt": ["grass", "road", "pavement"]
    },
    {
        "exp_name": "elbphilharmonie",
        "data_path": f"{MY_STORAGE}/Pexels/elbphilharmonie",
        "bldg_prompt": "Elbphilharmonie, that modernism red-brick and glass building",
        "gnd_prompt": ["water", "road", "pavement"]
    },
    {
        "exp_name": "krasna-horka-castle",
        "data_path": f"{MY_STORAGE}/Pexels/krasna-horka-castle",
        "bldg_prompt": "that castle building",
        "gnd_prompt": ["ground", "grass", "pavement"]
    }
]

# ================================================================================
# Main Function
# ================================================================================
if __name__ == "__main__":
    # Set HF token
    token_path = os.path.join(os.path.dirname(__file__), "SAM3", "hf_token.txt")
    set_hf_token_from_txt(token_path)

    # Starts here
    for scene in SCENES:
        # SAM3 -> scene['data_path']/sam3.log
        with stage_file_logger(scene["data_path"], "sam3.log"):
            sam_inference_a_scene(scene)

        # DA3 -> scene['data_path']/da3.log
        with stage_file_logger(scene["data_path"], "da3.log"):
            da3_inference_a_scene(scene)

        ## PlanarSplatting ====================
        run_planarsplatting(
            data_path=scene['data_path'],
            exp_name=scene['exp_name'],
            out_path="A3_progress/DA3FG2_split",
            conf_path="configs/DA3FG++big.conf",
            mask="bldg_masks"
        )
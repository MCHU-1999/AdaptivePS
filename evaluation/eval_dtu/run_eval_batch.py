"""
Batch evaluation of DTU scenes.

Run from the PlanarSplatting project root:
    python evaluation/eval_dtu/run_eval_batch.py

Outputs per-scene results.json into VIS_OUT_DIR/scanXX/results.json

You want to see as much White and faint pink as possible. 
Red means significant inaccuracy, 
Green means terrible outliers or huge holes,
Blue is just background context that doesn't affect your metrics.
"""
import os
import subprocess


# ==============================================================
# CONFIG
# ==============================================================
RESULTS_DIR = "Vanilla/DTU-Building"          # contains scanXX_DA3FG/ subdirs
VIS_OUT_DIR = "Vanilla/eval_results"  # where results go
DATASET_DIR = "/Users/mchu/Documents/TUD/Thesis/DTU_GT"
PLY_NAME    = "planar_mesh.ply"
SCALE       = 100.0   # meters → mm to match DTU GT

EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "eval.py")
# ==============================================================


def find_latest_run(scene_dir: str):
    """Return the path to the most recent timestamped sub-folder."""
    entries = sorted([
        e for e in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, e))
    ])
    return os.path.join(scene_dir, entries[-1]) if entries else None


if __name__ == "__main__":
    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(f"Results dir not found: {RESULTS_DIR}\n"
                                f"Make sure you are running from the project root (PlanarSplatting/).")

    scene_folders = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])
    print(f"Found {len(scene_folders)} scene(s) in {RESULTS_DIR}\n")

    for scene_folder in scene_folders:
        scene_dir = os.path.join(RESULTS_DIR, scene_folder)

        # e.g. "scan24_DA3FG" → scan_name="scan24", scan_num=24
        scan_name = scene_folder.split("_")[0]
        try:
            scan_num = int(scan_name.replace("scan", ""))
        except ValueError:
            print(f"[SKIP] Cannot parse scan number from '{scene_folder}'")
            continue

        run_dir = find_latest_run(scene_dir)
        if run_dir is None:
            print(f"[SKIP] {scene_folder}: no timestamped run dir found")
            continue

        ply_path = os.path.abspath(os.path.join(run_dir, PLY_NAME))
        if not os.path.exists(ply_path):
            print(f"[SKIP] {scene_folder}: PLY not found at {ply_path}")
            continue

        vis_out = os.path.join(VIS_OUT_DIR, scan_name)

        print(f"{'='*60}")
        print(f"  Scene:  {scene_folder}  (scan {scan_num})")
        print(f"  PLY:    {ply_path}")
        print(f"  Output: {vis_out}")

        cmd = [
            "python", EVAL_SCRIPT,
            "--data",       ply_path,
            "--scan",       str(scan_num),
            "--mode",       "mesh",
            "--dataset_dir", DATASET_DIR,
            "--vis_out_dir", vis_out,
            "--scale",      str(SCALE),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] {scene_folder} exited with code {result.returncode}")

    print("\nAll scenes evaluated.")

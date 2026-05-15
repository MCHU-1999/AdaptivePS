import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import numpy as np
from planarsplat.data_process.colmap_io import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
)
from planarsplat.utils.mesh_util import render_rgb, render_DNs_color
from PIL import Image
import trimesh


def read_colmap_cameras(colmap_dir):
    """Read COLMAP intrinsics and extrinsics, return (c2w poses, K matrices, H, W, names)."""
    # Intrinsics
    cam_bin = os.path.join(colmap_dir, "cameras.bin")
    cam_txt = os.path.join(colmap_dir, "cameras.txt")
    if os.path.exists(cam_bin):
        cameras = read_intrinsics_binary(cam_bin)
    elif os.path.exists(cam_txt):
        cameras = read_intrinsics_text(cam_txt)
    else:
        raise FileNotFoundError(f"No cameras file found in {colmap_dir}")

    # Extrinsics
    img_bin = os.path.join(colmap_dir, "images.bin")
    img_txt = os.path.join(colmap_dir, "images.txt")
    if os.path.exists(img_bin):
        images_meta = read_extrinsics_binary(img_bin)
    elif os.path.exists(img_txt):
        images_meta = read_extrinsics_text(img_txt)
    else:
        raise FileNotFoundError(f"No images file found in {colmap_dir}")

    poses, intrinsics, names = [], [], []
    H = W = None

    for img_id, img_meta in images_meta.items():
        q = img_meta.qvec
        t = img_meta.tvec
        r = qvec2rotmat(q)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)

        # Intrinsics — handle per-image or single camera
        cam_id = img_meta.camera_id if hasattr(img_meta, 'camera_id') else img_id
        camera = cameras.get(cam_id, next(iter(cameras.values())))
        fx, fy, cx, cy = camera.params[:4]
        K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], dtype=np.float32)
        intrinsics.append(K)

        if H is None:
            H, W = camera.height, camera.width

        names.append(os.path.splitext(img_meta.name)[0])   # stem without extension

    return poses, intrinsics, H, W, names

def find_latest_run(scene_dir):
    """Return the path to the most recent timestamped sub-folder inside scene_dir."""
    entries = sorted([
        e for e in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, e))
    ])
    if not entries:
        return None
    return os.path.join(scene_dir, entries[-1])


def render_scene(ply_path, colmap_dir):
    """Render RGB, depth and normals for one PLY using the COLMAP cameras.
    Outputs are saved next to the PLY file in rendered_rgb/ and rendered_dn/."""
    print(f"\n{'='*60}")
    print(f"  PLY:    {ply_path}")
    print(f"  Cams:   {colmap_dir}")

    mesh = trimesh.load(ply_path, force='mesh')
    print(f"  Mesh:   {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    poses, intrinsics, H, W, names = read_colmap_cameras(colmap_dir)
    print(f"  Cameras: {len(poses)} @ {W}x{H}")

    run_dir = os.path.dirname(ply_path)
    rgb_dir = os.path.join(run_dir, "rendered_rgb")
    dn_dir  = os.path.join(run_dir, "rendered_dn")

    # --- RGB first (does NOT mutate mesh colours) ---
    print(f"  Rendering RGB → rendered_rgb/")
    rgbs = render_rgb(mesh, poses, intrinsics, H, W)
    save_images(rgbs, names, rgb_dir, suffix="")

    # --- Depth + Normals (mutates vertex colours, must come after RGB) ---
    print(f"  Rendering Depth + Normals → rendered_dn/")
    depths, normals = render_DNs_color(mesh, poses, intrinsics, H, W)
    save_images(normals, names, dn_dir, suffix="_normal")
    save_images(depths,  names, dn_dir, suffix="_depth", as_depth=True)

    print(f"  Done.")


def save_images(images, names, out_dir, suffix, as_depth=False):
    """Save a list of numpy arrays as PNG files."""
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(images, names):
        out_path = os.path.join(out_dir, f"{name}{suffix}.png")
        if as_depth:
            # Scale to 16-bit PNG for lossless storage
            img_norm = (img / img.max() * 65535).astype(np.uint16) if img.max() > 0 else img.astype(np.uint16)
            Image.fromarray(img_norm).save(out_path)
        else:
            Image.fromarray(img.astype(np.uint8)).save(out_path)

def render_all(results_dir: str, ):
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    scene_dirs = sorted([
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    print(f"Found {len(scene_dirs)} scene(s) in {results_dir}")

    for scene_dir in scene_dirs:
        scene_folder = os.path.basename(scene_dir)   # e.g. "scan6_DA3FG"
        # Derive scan name: take the part before the first "_"
        scan_name = scene_folder.split("_")[0]        # e.g. "scan6"
        colmap_dir = os.path.join(DATA_BASE_DIR, scan_name, "DA3_colmap")

        run_dir = find_latest_run(scene_dir)
        if run_dir is None:
            print(f"[SKIP] No timestamped run found in {scene_dir}")
            continue

        ply_path = os.path.join(run_dir, PLY_NAME)
        if not os.path.exists(ply_path):
            print(f"[SKIP] PLY not found: {ply_path}")
            continue

        if not os.path.isdir(colmap_dir):
            print(f"[SKIP] COLMAP dir not found: {colmap_dir}")
            continue

        try:
            render_scene(ply_path, colmap_dir)
        except Exception as e:
            import traceback
            print(f"[ERROR] {scene_folder}: {e}")
            traceback.print_exc()

    print("\nAll scenes processed.")


# ==============================================================
# CONFIG — edit these paths before running
# ==============================================================
# MY_STORAGE      = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
MY_STORAGE      = "/Users/mchu/Documents/TUD/Thesis"
RESULTS_DIR     = "AdaptivePS/DTU-Building"      # root containing scanXX_DA3FG/ subdirs
DATA_BASE_DIR   = f"{MY_STORAGE}/DTU_ALL/Building"  # scanXX/ subdirs live here
PLY_NAME        = "planar_mesh.ply"
# ==============================================================


if __name__ == "__main__":
    render_all(RESULTS_DIR)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import numpy as np
from planarsplat.data_process.colmap_io import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text, qvec2rotmat
)
from PIL import Image
import trimesh
from tqdm import tqdm
import pyrender


###
# Class definition
class FastRenderer():
    def __init__(self, mesh, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene(bg_color=[1, 1, 1, 1])
        
        # Add mesh ONCE during initialization
        self.mesh_node = self.scene.add(mesh)
        
        # Add a placeholder camera node ONCE
        # Will update its pose and intrinsics later
        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0) # Placeholder
        self.cam_node = self.scene.add(self.cam, pose=np.eye(4))
        
        self.render_flags = pyrender.RenderFlags.FLAT

    def render_frame(self, height, width, intrinsics, pose):
        # Update viewport if it changed (usually static in a batch)
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        
        # Update Camera Intrinsics 
        # (Updating the camera object is faster than adding/removing)
        new_cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2], cy=intrinsics[1, 2],
            fx=intrinsics[0, 0], fy=intrinsics[1, 1]
        )
        self.scene.main_camera_node.camera = new_cam
        
        # Update Camera Pose
        self.scene.set_pose(self.cam_node, pose=self.fix_pose(pose))
        
        return self.renderer.render(self.scene, flags=self.render_flags)

    def fix_pose(self, pose):
        # Corrects OpenCV (Z-forward) to OpenGL (Z-backward)
        R = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def delete(self):
        self.renderer.delete()

###
# Below are functions for rendering experiment results
def render_rgb(mesh, poses, intrinsics, H, W):
    mesh_opengl = pyrender.Mesh.from_trimesh(mesh)
    renderer = FastRenderer(mesh_opengl, height=H, width=W)
    
    rendered_rgbs = []
    try:
        for pose, K in tqdm(zip(poses, intrinsics), total=len(poses)):
            rgb, depth = renderer.render_frame(H, W, K, pose)
            # RenderFlags.FLAT bypasses bg_color — fill background (depth==0) with white
            rgb[depth == 0] = 255
            rendered_rgbs.append(rgb)
    finally:
        renderer.delete()
    return rendered_rgbs

def render_DNs_color(mesh, poses, intrinsics, H, W):
    """
    Render both depths and normals (Normals in RGB space)
    """
    # Extract vertex normals and normalize them to RGB space
    # Formula: RGB = (Normals + 1) / 2
    normals = mesh.vertex_normals
    rgb_normals = (normals + 1.0) / 2.0 
    rgb_normals_uint8 = (rgb_normals * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = rgb_normals_uint8

    # Create Pyrender mesh
    mesh_opengl = pyrender.Mesh.from_trimesh(mesh)
    renderer = FastRenderer(mesh_opengl, height=H, width=W)
    
    rendered_depths = []
    rendered_normals = []
    try:
        for pose, K in tqdm(zip(poses, intrinsics), total=len(poses)):
            normal_pred, depth_pred = renderer.render_frame(H, W, K, pose)
            rendered_normals.append(normal_pred)
            rendered_depths.append(depth_pred)
    finally:
        renderer.delete()
    return rendered_depths, rendered_normals


###
#$ Function definition
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


def render_scene(ply_path, colmap_dir, scan_name, out_dir, indices=None):
    """Render RGB (and optionally D+N) for one PLY using the COLMAP cameras.
    Outputs go to:
      out_dir/rendered_rgb/{scan_name}/
      out_dir/rendered_dn/{scan_name}/

    Args:
        indices: optional list of integer indices into the camera list.
                 e.g. [1, 3] renders only names[1] and names[3].
                 If None (default), renders all cameras.
    """
    print(f"\n{'='*60}")
    print(f"  PLY:    {ply_path}")
    print(f"  Cams:   {colmap_dir}")

    mesh = trimesh.load(ply_path, force='mesh')
    print(f"  Mesh:   {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    poses, intrinsics, H, W, names = read_colmap_cameras(colmap_dir)
    print(f"  Cameras: {len(poses)} @ {W}x{H}")

    # Filter to requested indices if provided
    if indices is not None:
        n = len(poses)
        valid = [i for i in indices if i < n]
        skipped = [i for i in indices if i >= n]
        if skipped:
            print(f"  Skipping indices out of range (n={n}): {skipped}")
        poses      = [poses[i]      for i in valid]
        intrinsics = [intrinsics[i] for i in valid]
        names      = [names[i]      for i in valid]
        print(f"  Rendering subset: {valid} → {len(poses)} camera(s)")

    rgb_dir = os.path.join(out_dir, "rendered_rgb", scan_name)
    dn_dir  = os.path.join(out_dir, "rendered_dn",  scan_name)

    # --- RGB first (does NOT mutate mesh colours) ---
    print(f"  Rendering RGB → {rgb_dir}")
    rgbs = render_rgb(mesh, poses, intrinsics, H, W)
    save_images(rgbs, names, rgb_dir, suffix="")

    # # --- Depth + Normals (mutates vertex colours, must come after RGB) ---
    # print(f"  Rendering Depth + Normals → {dn_dir}")
    # depths, normals = render_DNs_color(mesh, poses, intrinsics, H, W)
    # save_images(normals, names, dn_dir, suffix="_normal")
    # save_images(depths,  names, dn_dir, suffix="_depth", as_depth=True)

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

def render_all(results_dir: str, out_dir: str, indices=None):
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    scene_dirs = sorted([
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    print(f"Found {len(scene_dirs)} scene(s) in {results_dir}")
    print(f"Output root: {out_dir}")

    for scene_dir in scene_dirs:
        scene_folder = os.path.basename(scene_dir)   # e.g. "scan6_DA3FG"
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
            render_scene(ply_path, colmap_dir, scan_name=scan_name, out_dir=out_dir, indices=indices)
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
RESULTS_DIR     = "AdaptivePS/TnT"         # root containing scanXX_DA3FG/ subdirs
DATA_BASE_DIR   = f"{MY_STORAGE}/TNT_GOF/TrainingSet"  # scanXX/ subdirs live here
PLY_NAME        = "planar_mesh.ply"
OUTPUT_DIR      = "AdaptivePS"                         # rendered_rgb/{scan} and rendered_dn/{scan} go here
# ==============================================================


if __name__ == "__main__":
    # INDICES = [0, 25, 45, 75, 100, 125, 150]
    INDICES = [i for i in range(0, 400, 25)]
    render_all(RESULTS_DIR, out_dir=OUTPUT_DIR, indices=INDICES)

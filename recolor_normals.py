import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Import colmap_io from the workspace
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from planarsplat.data_process.colmap_io import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat

def main():
    parser = argparse.ArgumentParser(description="Recolor local normal .npy files to global normal .png images.")
    parser.add_argument("--normal_dir", type=str, required=True, help="Directory containing .npy normal files")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Directory containing COLMAP sparse model (images.bin or images.txt)")
    args = parser.parse_args()

    # Load colmap extrinsics
    if os.path.exists(os.path.join(args.colmap_dir, "images.bin")):
        images = read_extrinsics_binary(os.path.join(args.colmap_dir, "images.bin"))
        print(f"Loaded {len(images)} images from images.bin")
    elif os.path.exists(os.path.join(args.colmap_dir, "images.txt")):
        images = read_extrinsics_text(os.path.join(args.colmap_dir, "images.txt"))
        print(f"Loaded {len(images)} images from images.txt")
    else:
        raise FileNotFoundError(f"Could not find images.bin or images.txt in {args.colmap_dir}")

    # Process each image in colmap
    for image_id in tqdm(images.keys(), desc="Recoloring normals"):
        image = images[image_id]
        # image.name is like "00000.jpg", we want "00000.npy"
        base_name = os.path.splitext(image.name)[0]
        
        # Sometimes images are in subdirectories, flatten or keep them depending on normal_dir structure
        # We will assume normals are stored as just the basename.npy or with the same relative path.
        # Let's try both: exact match vs basename only.
        npy_path = os.path.join(args.normal_dir, os.path.splitext(image.name)[0] + ".npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(args.normal_dir, os.path.basename(base_name) + ".npy")
            
        if not os.path.exists(npy_path):
            print(f"Skipping {image.name}: {npy_path} not found")
            continue
            
        # Load local normal, assumed shape: (3, H, W)
        normal = np.load(npy_path)
        
        # Determine if values are in [0, 1] or [-1, 1]
        # If min >= 0, it's likely [0, 1] encoding.
        if normal.min() >= 0.0 and normal.max() <= 1.0:
            normal_local = normal * 2.0 - 1.0
        else:
            normal_local = normal
            
        # Reshape to (H, W, 3)
        normal_local = np.transpose(normal_local, (1, 2, 0))
        H, W, _ = normal_local.shape
        
        # Flatten for matmul: shape (-1, 3)
        normal_local_flat = normal_local.reshape(-1, 3)
        
        # Get R_c2w from colmap (qvec is R_w2c)
        R_w2c = qvec2rotmat(image.qvec)
        R_c2w = R_w2c.T
        
        # Transform normals: N_global = N_local * R_c2w.T
        normal_global_flat = normal_local_flat @ R_c2w.T
        
        # Normalize to be safe
        norm = np.linalg.norm(normal_global_flat, axis=-1, keepdims=True)
        normal_global_flat = normal_global_flat / (norm + 1e-6)
        
        # Convert back to image shape (H, W, 3)
        normal_global = normal_global_flat.reshape(H, W, 3)
        
        # Convert to RGB [0, 255]
        normal_rgb = (normal_global + 1.0) / 2.0 * 255.0
        normal_rgb = np.clip(normal_rgb, 0, 255).astype(np.uint8)
        
        # Save as PNG
        out_path = os.path.splitext(npy_path)[0] + ".png"
        Image.fromarray(normal_rgb).save(out_path)

if __name__ == "__main__":
    main()

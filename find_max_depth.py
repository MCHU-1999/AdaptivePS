import os
import glob
import numpy as np
from tqdm import tqdm

def main():
    base_dir = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu/Pexels"
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Find all items in the base directory
    items = os.listdir(base_dir)
    
    global_max_depth = -1.0
    scene_max_depths = {}

    for item in items:
        scene_dir = os.path.join(base_dir, item)
        
        # Only process if it's a directory (excluding files)
        if not os.path.isdir(scene_dir):
            continue
            
        depth_dir = os.path.join(scene_dir, "DA3_depth")
        if not os.path.exists(depth_dir):
            continue
            
        # Find all .npy files in the DA3_depth folder
        npy_files = glob.glob(os.path.join(depth_dir, "*.npy"))
        
        if not npy_files:
            continue
            
        scene_max = -1.0
        
        # We can use tqdm to show progress within each scene, or across files
        print(f"Processing scene: {item} ({len(npy_files)} files)")
        for npy_file in tqdm(npy_files, leave=False):
            try:
                depth_map = np.load(npy_file)
                # handle potential NaN/Inf
                valid_mask = np.isfinite(depth_map)
                if np.any(valid_mask):
                    current_max = np.max(depth_map[valid_mask])
                    if current_max > scene_max:
                        scene_max = current_max
            except Exception as e:
                print(f"Error reading {npy_file}: {e}")
                
        if scene_max > -1.0:
            scene_max_depths[item] = scene_max
            if scene_max > global_max_depth:
                global_max_depth = scene_max
                
            print(f"  -> Max depth for {item}: {scene_max:.4f}")

    print("\n" + "="*50)
    print("SUMMARY OF MAX DEPTHS PER SCENE")
    print("="*50)
    for scene, max_d in sorted(scene_max_depths.items(), key=lambda x: x[1], reverse=True):
        print(f"{scene:<25}: {max_d:.4f}")
        
    print("-" * 50)
    print(f"GLOBAL MAXIMUM DEPTH: {global_max_depth:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

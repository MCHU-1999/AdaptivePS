import os
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# apptainer exec \
#     --containall \
#     -B /tudelft.net/:/tudelft.net/ \
#     --pwd /tudelft.net/staff-umbrella/Deep3D/mingchiehhu/PlanarSplatting \
#     /tudelft.net/staff-umbrella/Deep3D/mingchiehhu/containers/planarsplatting4.sif \
#     /opt/conda/envs/planarSplatting/bin/python find_max_depth.py

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
            
        # Sample 50 images to speed things up
        if len(npy_files) > 50:
            npy_files = random.sample(npy_files, 50)
            
        scene_valid_depths = []
        
        # We can use tqdm to show progress within each scene, or across files
        print(f"Processing scene: {item} ({len(npy_files)} files sampled)")
        for npy_file in tqdm(npy_files, leave=False):
            try:
                # Load depth
                depth_map = np.load(npy_file)
                
                # Derive mask path
                base_name = os.path.basename(npy_file)
                mask_name = base_name.replace('.npy', '.jpg')
                mask_file = os.path.join(scene_dir, "bldg_masks", mask_name)
                
                if not os.path.exists(mask_file):
                    continue
                    
                # Load mask
                mask_img = np.array(Image.open(mask_file))
                
                # If image has channels, take the first one or convert to grayscale
                if len(mask_img.shape) > 2:
                    mask_img = mask_img[..., 0]
                
                # Binary mask (assuming values are 0-255)
                valid_mask = (mask_img > 127)
                
                # Combine with finite depth mask
                valid_mask = valid_mask & np.isfinite(depth_map)
                
                if np.any(valid_mask):
                    scene_valid_depths.append(depth_map[valid_mask])
                    
            except Exception as e:
                print(f"Error reading {npy_file}: {e}")
                
        if len(scene_valid_depths) > 0:
            # Concatenate all valid depths for this scene
            all_depths = np.concatenate(scene_valid_depths)
            scene_95th = np.percentile(all_depths, 95)
            scene_max = np.max(all_depths)
            
            scene_max_depths[item] = {'95th': scene_95th, 'max': scene_max}
            if scene_95th > global_max_depth:
                global_max_depth = scene_95th
                
            print(f"  -> 95th-percentile: {scene_95th:.4f} | Max: {scene_max:.4f} for {item}")

    print("\n" + "="*65)
    print("SUMMARY OF DEPTHS PER SCENE (Valid Mask Area)")
    print("="*65)
    print(f"{'Scene':<25} | {'95th Percentile':<15} | {'Absolute Max':<15}")
    print("-" * 65)
    for scene, data in sorted(scene_max_depths.items(), key=lambda x: x[1]['95th'], reverse=True):
        print(f"{scene:<25} | {data['95th']:<15.4f} | {data['max']:<15.4f}")
        
    print("-" * 65)
    print(f"GLOBAL MAXIMUM (of the 95th percentiles): {global_max_depth:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()

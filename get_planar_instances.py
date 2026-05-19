import os
import argparse
import re

def find_latest_run(scene_dir):
    """Return the path to the most recent timestamped sub-folder inside scene_dir."""
    try:
        entries = sorted([
            e for e in os.listdir(scene_dir)
            if os.path.isdir(os.path.join(scene_dir, e))
        ])
        if not entries:
            return None
        return os.path.join(scene_dir, entries[-1])
    except Exception as e:
        return None

def get_final_planar_instances(log_file):
    """Parses train.log and returns the last found number of planar instances."""
    final_instances = None
    pattern = re.compile(r"number of planar instances = (\d+)")
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    final_instances = int(match.group(1))
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        
    return final_instances

def main(base_dir):
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Assuming structure: base_dir -> scan_folder -> run_folder(s) -> train.log
    scans = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"{'Scan':<15} {'Instances':<10}")
    print("-" * 80)
    
    for scan in scans:
        scan_dir = os.path.join(base_dir, scan)
        latest_run_dir = find_latest_run(scan_dir)
        
        if not latest_run_dir:
            print(f"{scan:<15} {'N/A':<10} No runs found")
            continue
            
        train_log = os.path.join(latest_run_dir, 'train.log')
        if not os.path.exists(train_log):
            print(f"{scan:<15} {'N/A':<10} No train.log found in {latest_run_dir}")
            continue
            
        instances = get_final_planar_instances(train_log)
        
        if instances is not None:
            # We only print the relative path from base_dir to make it cleaner
            rel_path = os.path.relpath(train_log, base_dir)
            print(f"{scan:<15} {instances:<10}")
        else:
            print(f"{scan:<15} {'N/A':<10} No instance count in {train_log}")


# ==============================================================
# CONFIG — edit these paths before running
# ==============================================================
# MY_STORAGE      = "/tudelft.net/staff-umbrella/Deep3D/mingchiehhu"
MY_STORAGE      = "/Users/mchu/Documents/TUD/Thesis"
# RESULTS_DIR     = "AdaptivePS/DTU-Building"
# RESULTS_DIR     = "Vanilla/DTU-Building"
RESULTS_DIR     = "Vanilla/TnT"
# ==============================================================

if __name__ == "__main__":
    main(base_dir=RESULTS_DIR)


## Baseline
# scan6   150       
# scan9   142 
# scan14  152       
# scan15  141       
# scan16  238       
# scan17  138       
# scan18  169       
# scan19  151       
# scan21  223       
# scan22  231       
# scan23  147       
# scan24  140       
# scan28  190       
# scan29  184       
# scan43  190       
# scan44  213       
# scan46  122       

## AdaptivePS
# scan6     94        
# scan9     135  
# scan14    121       
# scan15    102       
# scan16    118       
# scan17    191       
# scan18    156       
# scan19    63        
# scan21    179       
# scan22    174       
# scan23    176       
# scan24    82        
# scan28    127       
# scan29    127       
# scan43    121       
# scan44    179       
# scan46    134       

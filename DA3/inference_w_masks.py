import torch, sys
import numpy as np
from depth_anything_3.api import DepthAnything3
import argparse
from utils import read_dataset, synthesize_intrinsics


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_dir", type=str, default='path/to/colmap/data', help='path of input colmap data')
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    # Setup model and device
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    dataset = read_dataset(DATA_DIR)

    # Create synthetic intrinsics for outdoor scene (75° horizontal FOV)
    K_synthetic = synthesize_intrinsics(dataset.width, dataset.height, fov_deg=75)
    intrinsics = np.stack([K_synthetic] * dataset.N, axis=0)

    # Export depth data and 3D visualization
    prediction = model.inference(
        image=dataset.img_paths_list,
        extrinsics=None,
        intrinsics=None,
        export_dir=DATA_DIR,
        export_format="planarsplatting-colmap",
        process_res=420,
        # process_res=840,
        process_res_method="upper_bound_resize",
        export_kwargs={
            "planarsplatting": {
                "img_name_list": dataset.img_name_list,
                "img_res": [dataset.height, dataset.width]
            }
        },
        show_cameras=False,
        # show_cameras=True,
        conf_thresh_percentile=40,
        bldg_mask_paths=dataset.bldg_mask_paths,
        gnd_mask_paths=dataset.gnd_mask_paths
    )

    # prediction.processed_images : [N, H, W, 3] uint8   array
    print_err("processed_images.shape: ", prediction.processed_images.shape)

    # prediction.depth            : [N, H, W]    float32 array
    print_err("depth.shape: ", prediction.depth.shape)  

    # prediction.conf             : [N, H, W]    float32 array
    print_err("conf.shape: ", prediction.conf.shape)  

    # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    print_err("extrinsics.shape: ", prediction.extrinsics.shape)
    
    # prediction.intrinsics       : [N, 3, 3]    float32 array
    print_err("intrinsics.shape: ", prediction.intrinsics.shape)
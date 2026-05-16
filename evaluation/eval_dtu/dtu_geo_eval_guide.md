# DTU Geometry Evaluation — How to Run `eval.py`

This script computes the Chamfer distance between your reconstructed mesh and the
official DTU ground truth point clouds.

**Assumption:** your mesh is already in the DTU world coordinate system.
No registration or coordinate transform is applied.

---

## Prerequisites

### 1. Your Reconstruction

A single `.ply` file containing the reconstructed mesh (or point cloud) for one scan,
already in the DTU world coordinate space.

```
/path/to/your/scan24.ply
```

### 2. Official DTU Evaluation Dataset

Download the evaluation data from:
https://roboimagedata.compute.dtu.dk/?page_id=36

The expected folder structure is:

```
/path/to/dtu_eval/
├── ObsMask/
│   ├── ObsMask24_10.mat      ← 3D observation mask for scan 24
│   ├── Plane24.mat           ← ground plane for scan 24
│   ├── ObsMask37_10.mat
│   ├── Plane37.mat
│   └── ...                   ← one pair per scan ID
└── Points/
    └── stl/
        ├── stl024_total.ply  ← GT point cloud for scan 24
        ├── stl037_total.ply
        └── ...               ← zero-padded 3-digit scan IDs
```

---

## Usage

```bash
cd /path/to/PGSR/scripts/eval_dtu

python eval.py \
    --data         /Users/mchu/Documents/TUD/Thesis/PlanarSplatting/AdaptivePS/DTU-Building/scan24_DA3FG/2026_05_15_12_16_38/planar_mesh.ply \
    --scan         24 \
    --mode         mesh \
    --dataset_dir  /Users/mchu/Documents/TUD/Thesis/DTU_GT \
    --vis_out_dir  results/scan24
```

The output directory (`--vis_out_dir`) will be created automatically if it does not exist.

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `data_in.ply` | Path to your reconstructed `.ply` file |
| `--scan` | `1` | Integer scan ID (e.g. `24`, `37`, `110`) |
| `--mode` | `mesh` | `mesh` to sample points from a mesh, `pcd` if input is already a point cloud |
| `--dataset_dir` | `.` | Root of the official DTU evaluation dataset |
| `--vis_out_dir` | `.` | Directory to write `results.json` and error visualizations |
| `--downsample_density` | `0.2` | Voxel size used when downsampling the sampled point cloud |
| `--patch_size` | `60` | Extra padding (mm) added around the bounding box when masking |
| `--max_dist` | `20` | Points further than this (mm) from the nearest GT point are excluded from the mean |
| `--visualize_threshold` | `10` | Distance threshold (mm) for the error colormap in visualization PLYs |

> **Note:** The defaults match the official DTU evaluation protocol used in most papers.
> Do not change them unless you have a specific reason.

---

## What It Computes

The script computes the standard DTU Chamfer distance metrics:

| Metric | Key in JSON | Description |
|---|---|---|
| **Accuracy** | `mean_d2s` | Mean distance from your reconstruction to the GT point cloud. Measures geometric precision. |
| **Completeness** | `mean_s2d` | Mean distance from the GT point cloud to your reconstruction. Measures how much of the surface was captured. |
| **Overall** | `overall` | `(mean_d2s + mean_s2d) / 2`. The primary reported metric. |

Lower is better for all three. Units are millimetres in DTU world scale.

---

## Output Files

After running, `--vis_out_dir` will contain:

```
/path/to/output/scan24/
├── results.json              ← Chamfer distance metrics
├── vis_024_d2s.ply           ← your mesh coloured by error (reconstruction → GT)
└── vis_024_s2d.ply           ← GT coloured by error (GT → reconstruction)
```

`results.json` example:
```json
{
    "mean_d2s": 0.42,
    "mean_s2d": 0.61,
    "overall":  0.51
}
```

---

## How the Masking Works

`eval.py` applies two levels of filtering to your point cloud before computing distances,
in order to only evaluate geometry that was actually observable:

1. **Bounding box** (`BB` from `ObsMask*.mat` + `--patch_size` padding): removes points
   far outside the scene region of interest.
2. **3D observation mask** (`ObsMask` voxel grid): removes points in regions that were
   never visible from any camera. This ensures you are not penalized for missing geometry
   that no camera could ever see.

These filters are applied **only to Accuracy** (`mean_d2s`). For Completeness (`mean_s2d`),
the GT points are filtered only by the ground plane (`Plane*.mat`) to remove the table surface.

# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
import csv
from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from evaluation import EvaluateHisto, compute_chamfer
from util import make_dir
from plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # this is for groundtruth pointcloud, we can use it
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # this crop file is also w.r.t the groundtruth pointcloud, we can use it. 
    # Otherwise we have to crop the estimated pointcloud by ourself
    cropfile = os.path.join(dataset_dir, scene + ".json")
    # this is not so necessary
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None

    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.remove_unreferenced_vertices()
    # pcd = mesh.sample_points_uniformly(number_of_points=12800000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    # pcd = o3d.io.read_point_cloud(ply_path)
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    traj_to_register = read_trajectory(traj_path)
    gt_traj_col = read_trajectory(colmap_ref_logfile)

    trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                gt_traj_col, gt_trans, scene)

    # Refine alignment by using the actual GT and MVS pointclouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau

    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":

    BASE = "/Users/mchu/Documents/TUD/Thesis"
    RUNS = [
        "Allnone",
        "No1mesh",
        "Normalswap",
        "Nosplit",
        "Notrim",
        "Only1mesh",
        "Onlysplit",
        "Onlytrim"
    ]

    DATASET_DIR = f"{BASE}/TNT_GOF/TrainingSet/Barn"
    TRAJ_PATH = f"{BASE}/TNT_GOF/TrainingSet/Barn/DA3_colmap/DA3.log"
    PLY_PATHS = [f"{BASE}/PlanarSplatting/Ablation_tnt/{dir}/planar_mesh.ply" for dir in RUNS]
    OUT_DIRS = [f"{BASE}/PlanarSplatting/evaluation/eval_tnt/ablation/{dir}" for dir in RUNS]

    for ply_path, out_dir in zip(PLY_PATHS, OUT_DIRS):
        run_evaluation(
            dataset_dir=DATASET_DIR,
            traj_path=TRAJ_PATH,
            ply_path=ply_path,
            out_dir=out_dir,
        )

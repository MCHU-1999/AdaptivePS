import os
import math
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
# from pytorch3d.ops import knn_points
from pykeops.torch import LazyTensor
import random
import itertools
from utils.model_util import quat_to_rot
from utils.plot_util import plot_rectangle_planes

def merge_plane(
    net, 
    coarse_mesh_o3d,    # This mesh is fused using voxel_length=dataset.voxel_length
    plane_ins_id: Optional[List]=None, 
    normal_angle_thresh: float=25,
    dist_thresh: float=0.05, 
    mesh_dist_thresh: float=0.01,
    mesh_dist_thresh_2: float=0.015,
    space_resolution: float=0.02,   # This decides the point sampling (and the final mesh) resolution
    voxel_size: float=0.01, # This is not used to fuse mesh, instead it's used to determine connectivity 
    min_pts_num: int=25,    # This is used to remove small planes
    view_info_list=None,
    H: Optional[int]=None,
    W: Optional[int]=None,
    trim_enabled: bool=False,
):
    torch.use_deterministic_algorithms(False)
    min_pts_num = max((0.1/space_resolution)**2, min_pts_num)

    ## get parameters of 3D plane primitives
    with torch.no_grad():
        plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, _, _ = net.planarSplat.get_plane_geometry()
        plane_center = plane_center.detach()
    
    ## sample points from the 3D plane primitives
    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]).cuda() + 1
    pts_original, pts_ins_assignment_original, pts_normal_original, faces_original = sample_pts_from_GivenPlanePrim(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        space_resolution=space_resolution,
        plane_ins_id=plane_ins_id)
    
    ## calculate pts2mesh distance
    dist_pts2mesh_original = calculate_pts2mesh_dist(pts_original, coarse_mesh_o3d)

    ## calculate masked pts assignment
    pts_ins_assignment_masked = pts_ins_assignment_original.clone()
    pts_ins_assignment_masked[dist_pts2mesh_original > mesh_dist_thresh] = 0

    ## mask the bg points as well
    if trim_enabled and view_info_list is not None and H is not None and W is not None:
        pts_ins_assignment_masked = mask_points_in_bg(
            pts_world=pts_original,
            pts_ins_assignment=pts_ins_assignment_masked,
            view_info_list=view_info_list,
            H=H,
            W=W,
            max_bg_ratio=0.4,
        )

    ## split planes into different group via normal 
    # NG almost certainly means "normal grouped"
    pts_ins_assignment_masked_NG = group_plane_via_normal(
        pts_normal_original, # S, 3
        pts_ins_assignment_masked, # S
        normal_angle_thresh=normal_angle_thresh,
    )
    pts_ins_assignment_masked_NG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG).int()
    ## update pts' normal
    pts_normal_updated = update_pts_normal(pts_ins_assignment_masked_NG, pts_normal_original, use_median=True)
    ## adjust pts for each plane (***optinal***)
    if True:
        pts_updated = move_pts_onto_plane_simple(pts_original.clone(), pts_normal_updated, pts_ins_assignment_masked)
    else:
        pts_updated = pts_original.clone()

    ## check cc
    # CC almost certainly means Connected Components
    pts_ins_assignment_masked_NG_cc = torch.zeros_like(pts_ins_assignment_masked_NG)
    NG_ids = pts_ins_assignment_masked_NG.unique()
    for ngid in NG_ids:         
        if ngid == 0:
            continue   
        gmask = pts_ins_assignment_masked_NG == ngid
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked * gmask.float()
        pts_ins_assignment_masked_tmp = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()
        pts_ins_assignment_masked_tmp_cc = check_cc(
            pts_updated, # S, 3
            pts_ins_assignment_masked_tmp, # S
            adj_ratio_threshold=0.0, 
            adj_count_threshold=5,
            voxel_size=voxel_size,
        )
        last_id = pts_ins_assignment_masked_NG_cc.max() + 1
        pts_ins_assignment_masked_NG_cc[gmask] = pts_ins_assignment_masked_tmp_cc[gmask] + last_id
    pts_ins_assignment_masked_NG_cc= get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc).int()

    ## check distdance
    pts_ins_assignment_masked_tmp = torch.zeros_like(pts_ins_assignment_masked)
    start_ins_id = 1
    for group_id in pts_ins_assignment_masked_NG_cc.unique():
        if group_id == 0:
            continue
        pts_ins_assignment_masked_cur = find_best_assignments_for_each_group(
            group_id, 
            pts_updated,
            pts_normal_updated, 
            pts_ins_assignment_masked_NG_cc, 
            pts_ins_assignment_masked, 
            start_ins_id,
            dist_thres=dist_thresh)
        assert pts_ins_assignment_masked_tmp[pts_ins_assignment_masked_cur>0].sum() == 0
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked_tmp + pts_ins_assignment_masked_cur
        start_ins_id = pts_ins_assignment_masked_tmp.max()+1
    pts_ins_assignment_masked_NG_cc_DG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()

    ## merge floor again
    pts_ins_assignment_masked_NG_cc_DG_mf = merge_coplanar(
        pts_updated, # S, 3
        pts_normal_updated, # S, 3
        pts_ins_assignment_masked_NG_cc_DG, # S
        voxel_size=voxel_size,
        normal_angle_thresh=normal_angle_thresh,
        dist_thresh=dist_thresh
    )
    pts_ins_assignment_masked_NG_cc_DG_mf = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc_DG_mf).int()
    pts_ins_assignment_final = pts_ins_assignment_masked_NG_cc_DG_mf.clone()

    ## move points onto plane
    pts_updated = move_pts_onto_plane_simple(pts_updated.clone(), pts_normal_updated, pts_ins_assignment_final)

    ## remove small planes
    min_ins_pts = min_pts_num
    count = 0
    for label in pts_ins_assignment_final.unique():
        if label == 0:
            continue
        mask = pts_ins_assignment_final == label    # binary mask
        if mask.sum() < min_ins_pts:
            pts_ins_assignment_final[mask] = 0
        else:
            count += 1
    logger.info("number of planar instances = %d"%(count))
    pts_ins_assignment_final = get_continues_pts_ins_assignment(pts_ins_assignment_final).int()

    # update
    if mesh_dist_thresh_2 > mesh_dist_thresh:
        pts_ins_assignment_final_tmp = torch.zeros_like(pts_ins_assignment_final)
        for label in pts_ins_assignment_final.unique():
            if label == 0:
                continue
            mask = pts_ins_assignment_final == label
            original_ids = pts_ins_assignment_original[mask].unique()
            for oid in original_ids:
                mask2 = (pts_ins_assignment_original == oid) & (dist_pts2mesh_original <= mesh_dist_thresh_2)
                pts_ins_assignment_final_tmp[mask2] = label
        pts_ins_assignment_final = pts_ins_assignment_final_tmp.clone()
        pts_ins_assignment_final = get_continues_pts_ins_assignment(pts_ins_assignment_final).int()

    plane_ins_id_new = get_planeInsId_from_ptsInsAssignment(plane_normal.shape[0], pts_ins_assignment_original, pts_ins_assignment_final)
    planar_mesh, mesh_pts = build_planar_mesh_for_eval(
        pts_updated, 
        pts_normal_updated, 
        pts_ins_assignment_final, 
        faces_original, 
        return_pts=True
    )
    
    return planar_mesh, plane_ins_id_new

def mask_points_in_bg(
    pts_world,
    pts_ins_assignment,
    view_info_list,
    H,
    W,
    max_bg_ratio=0.4
):
    with torch.no_grad():
        device = pts_world.device
        keep_label = pts_ins_assignment.clone()

        valid_pts = keep_label > 0
        if valid_pts.sum() == 0:
            return keep_label

        pts = pts_world
        ones = torch.ones((pts.shape[0], 1), device=device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=1)  # S,4

        visible_count = torch.zeros((pts.shape[0],), device=device, dtype=torch.float32)
        bg_count = torch.zeros((pts.shape[0],), device=device, dtype=torch.float32)

        for view_info in view_info_list:
            # pose is c2w, so inverse gives w2c
            w2c = torch.inverse(view_info.pose)  # 4,4
            K = view_info.intrinsic[:3, :3]      # 3,3

            cam = (pts_h @ w2c.T)[:, :3]  # S,3
            z = cam[:, 2]
            z_ok = z > 1e-6

            x = cam[:, 0]
            y = cam[:, 1]

            u = (K[0, 0] * (x / (z + 1e-8)) + K[0, 2]).round().long()
            vv = (K[1, 1] * (y / (z + 1e-8)) + K[1, 2]).round().long()

            in_img = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

            # Observed in image
            obs = z_ok & in_img & valid_pts
            # Unseen but in front of camera and out of frame
            oof = z_ok & (~in_img) & valid_pts

            if (obs.sum() == 0) and (oof.sum() == 0):
                continue

            if obs.sum() > 0:
                fg_map = view_info.fg_mask.view(H, W).bool()
                is_bg = ~fg_map[vv[obs], u[obs]]

                obs_idx = torch.where(obs)[0]
                visible_count[obs_idx] += 1.0
                bg_count[obs_idx] += is_bg.float()

        bg_ratio = bg_count / (visible_count + 1e-6)
        remove = valid_pts & (bg_ratio > max_bg_ratio)
        keep_label[remove] = 0

        logger.info(
            f"bg-depth point mask removed {int(remove.sum().item())} points "
            f"({int(valid_pts.sum().item())} valid before)"
        )
        return keep_label

def merge_coplanar(
    pts, 
    pts_normal, 
    pts_ins_assignment_masked,
    normal_angle_thresh: float=15,
    dist_thresh: float=0.05,
    voxel_size: float=0.02
):
    normal_cos_thresh = math.cos(normal_angle_thresh / 180.0 * np.pi)

    unique_labels = pts_ins_assignment_masked.unique()
    max_label_num = len(unique_labels) - 1 if unique_labels[0] == 0 else len(unique_labels)

    mean_normal = torch.zeros((max_label_num, 3), device="cuda")
    mean_offset = torch.zeros((max_label_num, 1), device="cuda")
    mean_proj_dist = torch.ones((max_label_num, max_label_num), device="cuda") * 100.0

    used_labels = []
    pts_list = []

    count = 0
    for label in tqdm(unique_labels, total=len(unique_labels)):
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)

        plane_pts = pts[mask]
        plane_normal = torch.median(pts_normal[mask], dim=0)[0]
        normal = plane_normal / (torch.norm(plane_normal) + 1e-10)
        offset = -torch.mean((plane_pts * normal[None, :]).sum(-1), dim=0)

        mean_normal[count] = normal
        mean_offset[count] = offset
        pts_list.append(plane_pts.clone())
        count += 1

    assert len(used_labels) == count
    if count == 1:
        out = torch.zeros_like(pts_ins_assignment_masked)
        out[pts_ins_assignment_masked == used_labels[0]] = 1
        return out

    count = 0
    for label in tqdm(unique_labels, total=len(unique_labels)):
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        plane_pts = pts[mask]
        dist = (plane_pts.reshape(1, -1, 3) * mean_normal.reshape(-1, 1, 3)).sum(-1) + mean_offset.reshape(-1, 1)
        mean_proj_dist[count] = dist.abs().mean(dim=-1)
        count += 1

    normal_mask_nxn = (mean_normal[None, :] * mean_normal[:, None]).sum(2) > normal_cos_thresh
    normal_mask_nxn = normal_mask_nxn | normal_mask_nxn.t()

    dist_mask_nxn = mean_proj_dist < dist_thresh
    dist_mask_nxn = dist_mask_nxn | dist_mask_nxn.t()

    adj_mask_nxn = calculate_adj_mask(
        pts_list, 
        adj_ratio_threshold=0, 
        adj_count_threshold=1, 
        voxel_size=voxel_size
    )
    adj_mask_nxn = adj_mask_nxn | adj_mask_nxn.t()

    final_mask = adj_mask_nxn & dist_mask_nxn & normal_mask_nxn
    final_mask.fill_diagonal_(True)

    merged_groups = gpu_merge_overlapped_planes(final_mask)

    out = torch.zeros_like(pts_ins_assignment_masked)
    used_labels = torch.stack(used_labels).squeeze()
    new_ins_id = 1
    for group in merged_groups:
        group = torch.tensor(group).long()
        cur_labels = used_labels[group]
        for old_ins_id in cur_labels:
            out[pts_ins_assignment_masked == old_ins_id] = new_ins_id
        new_ins_id += 1
    return out

def find_best_assignments_for_each_group(group_id, pts_tensor, pts_normal_tensor, pts_group_assignment_tensor, pts_primitive_assignment_tensor, start_ins_id, dist_thres=0.1):
    pts_group_assignment_tensor = pts_group_assignment_tensor.clone()
    pts_primitive_assignment_tensor = pts_primitive_assignment_tensor.clone()
    pts_ins_assignment_tensor = torch.zeros_like(pts_group_assignment_tensor)
    
    pts_activate_flag = torch.zeros_like(pts_group_assignment_tensor)
    pts_activate_flag[pts_group_assignment_tensor==group_id] = 1

    activate_prim_IDs = pts_primitive_assignment_tensor[pts_group_assignment_tensor==group_id].unique()
    activate_prim_IDs = {pid.item(): True for pid in activate_prim_IDs}

    cur_ins_id = start_ins_id
    if True:
        for ite in range(len(activate_prim_IDs)):
            valid_pts_mask = pts_activate_flag > 0
            if valid_pts_mask.sum() == 0:
                break
            valid_pts = pts_tensor[valid_pts_mask]

            best_inlier_pts_mask = None
            best_inlier_num = 0
            best_prim_id = 0  # 0 means non-plane
            # find best prim id
            for prim_id, prim_flag in activate_prim_IDs.items():
                if not prim_flag:
                    continue
                prim_pts_mask = pts_primitive_assignment_tensor==prim_id
                assert pts_activate_flag[prim_pts_mask].min() > 0 
                prim_pts = pts_tensor[prim_pts_mask]
                prim_pts_normal = pts_normal_tensor[prim_pts_mask]
                prim_normal = F.normalize(prim_pts_normal.mean(dim=0), dim=-1)
                prim_offset = -torch.mean((prim_pts * prim_normal[None, :]).sum(-1), dim=0)
                # prim_offset = -torch.median((prim_pts * prim_normal[None, :]).sum(-1), dim=0)[0]

                pts2prim_dist = (valid_pts.reshape(-1, 3) * prim_normal.reshape(1, 3)).sum(-1) + prim_offset.reshape(1)
                pts2prim_dist = pts2prim_dist.abs()
                inlier_pts_mask = pts2prim_dist < dist_thres
                inlier_num = inlier_pts_mask.sum()
                if inlier_num > best_inlier_num:
                    best_inlier_num = inlier_num
                    best_inlier_pts_mask = inlier_pts_mask
                    best_prim_id = prim_id
            
            # prim assignment
            if best_prim_id > 0:
                best_inlier_primIDs = (pts_primitive_assignment_tensor[valid_pts_mask][best_inlier_pts_mask]).unique()
                for pid in best_inlier_primIDs:
                    pts_ins_assignment_tensor[pts_primitive_assignment_tensor==pid] = cur_ins_id
                    activate_prim_IDs[pid.item()] = False
                    pts_activate_flag[pts_primitive_assignment_tensor==pid] = 0
                cur_ins_id += 1
    return pts_ins_assignment_tensor

def build_planar_mesh_for_eval(pts, pts_normal, pts_ins_assignment, faces, move_pts_on=True, return_pts=False):
    import trimesh
    # ------------------------------------------------------------------- build planar mesh
    pts_ins_assignment_np = pts_ins_assignment.cpu().numpy()
    faces = faces.cpu().numpy()
    color_vis = random_color(np.unique(pts_ins_assignment_np).max().item()+100)
    colorMap_vis = color_vis(np.unique(pts_ins_assignment_np).max().item()+10)

    # -------------------------------------------------------------------- label non-plane faces
    face_ass = pts_ins_assignment_np[faces.reshape(-1)].reshape(-1, 3)
    face_invalid_mask = (face_ass[:,0] == 0).astype(np.int64) + (face_ass[:,1] == 0).astype(np.int64) + (face_ass[:,2] == 0).astype(np.int64)
    face_invalid_mask = face_invalid_mask == 3

    # -------------------------------------------------------------------- update pts color
    face_max_ass = face_ass.max(axis=-1).reshape(-1, 1)
    face_max_ass = np.repeat(face_max_ass, 3, axis=-1)
    pts_ins_assignment_np[faces[~face_invalid_mask]] = face_max_ass[~face_invalid_mask]
    pts_color = (colorMap_vis[pts_ins_assignment_np]).astype(np.uint8)

    # -------------------------------------------------------------------- move points
    if move_pts_on:
        pts_ins_assignment_tensor = torch.from_numpy(pts_ins_assignment_np).cuda()
        pts_plane = move_pts_onto_plane_simple(pts.clone(), pts_normal, pts_ins_assignment_tensor)
        vertices = pts_plane.cpu().numpy()
    else:
        vertices = pts.cpu().numpy()
        pts_plane = pts
        logger.warning("using original points....")

    # -------------------------------------------------------------------- convert to trimesh
    normals_np = pts_normal.cpu().numpy()   # per-vertex plane normals (S, 3)
    mesh = trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=pts_color,
        vertex_normals=normals_np,
        vertex_attributes={'pts_ins_assignment': pts_ins_assignment_np},
        process=False
    )
    
    # -------------------------------------------------------------------- remove non-plane faces
    mesh.update_faces(~face_invalid_mask)
    mesh.remove_unreferenced_vertices()

    if return_pts:
        return mesh, pts_plane
    else:
        return mesh
    
class random_color(object):
    def __init__(self, color_num=10000):
        num_of_colors=color_num
        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(num_of_colors)]

    def __call__(self, ret_n = 10):
        assert len(self.colors) > ret_n
        ret_color = np.zeros([ret_n, 3])
        for i in range(ret_n):
            hex_color = self.colors[i][1:]
            ret_color[i] = np.array([int(hex_color[j:j + 2], 16) for j in (0, 2, 4)])
        ret_color[0] *= 0
        return ret_color
        
def get_planeInsId_from_ptsInsAssignment(num_plane, pts_ins_assignment_original, pts_ins_assignment_updated):
    plane_ins_id = torch.zeros(num_plane).cuda()
    new_ids = pts_ins_assignment_updated.unique()
    for nid in new_ids:
        if nid == 0:
            continue
        pts_mask = pts_ins_assignment_updated == nid
        assert pts_mask.sum() > 0
        prim_ids = pts_ins_assignment_original[pts_mask].unique() - 1
        plane_ins_id[prim_ids] = nid.cpu().item()
    return plane_ins_id

def calculate_adj_mask(points_list, adj_ratio_threshold=0.05, adj_count_threshold=10, voxel_size=0.1):
    all_points = torch.cat(points_list)
    device = all_points.device
    plane_ids = torch.cat([torch.full((pts.size(0),), i, device=device, dtype=torch.long) 
                        for i, pts in enumerate(points_list)])
    
    base_voxels = torch.floor(all_points / voxel_size).long()
    offsets = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), 
                        device=device, dtype=torch.long)
    expanded_voxels = (base_voxels.unsqueeze(1) + offsets).view(-1, 3)
    expanded_plane_ids = plane_ids.repeat_interleave(len(offsets))
    
    combined = torch.cat([expanded_voxels, expanded_plane_ids.unsqueeze(1)], dim=1)
    unique_combined, _ = torch.unique(combined, dim=0, return_inverse=True)
    unique_voxels = unique_combined[:, :3]
    unique_planes = unique_combined[:, 3]
    
    unique_voxel_list, inverse_idx = torch.unique(unique_voxels, dim=0, return_inverse=True)
    
    num_voxels = unique_voxel_list.size(0)
    num_planes = len(points_list)
    indices = torch.stack([inverse_idx, unique_planes], dim=0)
    voxel_plane_matrix = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1), device=device, dtype=torch.float32),
        (num_voxels, num_planes)
    ).coalesce()
    
    dense_matrix = voxel_plane_matrix.to_dense()
    
    plane_voxel_counts = dense_matrix.sum(dim=0)  
    
    adj_count_matrix = torch.mm(dense_matrix.t(), dense_matrix) 
    
    ratio_AB = adj_count_matrix / (plane_voxel_counts.unsqueeze(1) + 1e-8)
    ratio_BA = adj_count_matrix / (plane_voxel_counts.unsqueeze(0) + 1e-8) 
    max_ratio = torch.maximum(ratio_AB, ratio_BA)
    
    adj_matrix = (max_ratio >= adj_ratio_threshold) & (adj_count_matrix >= adj_count_threshold)
    adj_matrix.fill_diagonal_(True)
    
    return adj_matrix.to(torch.bool)

def check_cc(
    pts,  # S, 3
    pts_ins_assignment_masked,  # S
    adj_ratio_threshold=0.05,
    adj_count_threshold=10,
    voxel_size=0.1,
):
    """
    Split labels into connected components using voxel-overlap adjacency only.
    Returns relabeled per-point assignments (0 stays non-plane).
    """
    unique_labels = pts_ins_assignment_masked.unique()

    # labels should be continuous and start from 0 or 1
    if unique_labels[0] == 0:
        assert unique_labels[-1] == len(unique_labels) - 1
    else:
        assert unique_labels[-1] == len(unique_labels)

    used_labels = []
    pts_list = []

    for label in unique_labels:
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)
        pts_list.append(pts[mask].clone())

    count = len(used_labels)

    # nothing to merge
    if count == 0:
        return torch.zeros_like(pts_ins_assignment_masked)

    if count == 1:
        out = torch.zeros_like(pts_ins_assignment_masked)
        out[pts_ins_assignment_masked == used_labels[0]] = 1
        return out

    # overlap-based adjacency graph
    adj_mask_nxn = calculate_adj_mask(
        pts_list,
        adj_ratio_threshold=adj_ratio_threshold,
        adj_count_threshold=adj_count_threshold,
        voxel_size=voxel_size,
    )
    adj_mask_nxn = adj_mask_nxn | adj_mask_nxn.t()

    final_mask = adj_mask_nxn
    final_mask.fill_diagonal_(True)

    merged_groups = gpu_merge_overlapped_planes(final_mask)

    # map connected components back to original labels
    out = torch.zeros_like(pts_ins_assignment_masked)
    used_labels = torch.stack(used_labels).squeeze()
    new_ins_id = 1
    for group in merged_groups:
        group = torch.tensor(group, device=used_labels.device).long()
        cur_labels = used_labels[group]
        for old_ins_id in cur_labels:
            out[pts_ins_assignment_masked == old_ins_id] = new_ins_id
        new_ins_id += 1

    return out

def gpu_merge_overlapped_planes(M):
    M = M | M.t()
    adj = M.to(torch.float32)
    n = adj.size(0)
    label = torch.eye(n, dtype=torch.float32, device=M.device)
    while True:
        new_label = label @ adj
        new_label = (new_label > 0).to(torch.float32)
        if torch.allclose(new_label, label):
            break
        label = new_label
    mask = label.bool()
    unique_masks = torch.unique(mask, dim=0, sorted=False)
    groups = []
    for m in unique_masks:
        group = torch.where(m)[0].cpu().tolist()
        groups.append(group)
    return groups

def move_pts_onto_plane_simple(ori_pts, pts_normal, per_point_plane_assignment):
    '''
    per_point_plane_assignment: 0 means non-plane / invalid plane
    '''
    dis_list = []
    max_label = per_point_plane_assignment.max() + 1
    move_dists = []
    for label in tqdm(range(max_label), desc='moving points to plane instance...'):
        if label > 0:
            mask = per_point_plane_assignment == label
            if mask.sum() == 0:
                continue
            # compute plane parameters and move all points onto the plane
            sample_pts = ori_pts[mask].clone()
            plane_normal = torch.median(pts_normal[mask], dim=0)[0]

            plane_offset = -torch.median((sample_pts @ plane_normal))

            dist = (sample_pts @ plane_normal) + plane_offset
            projected_points = sample_pts - plane_normal * dist[:, None]

            ori_pts[mask] = projected_points

            move_dists.append(dist)
    
    move_dists = torch.cat(move_dists, dim=0)
    avg_mv = torch.mean(move_dists)
    print(f"avg moving distance = {avg_mv}")
    print(f"max moving distance = {move_dists.abs().max()}")
    print(f"moving distance > 0.2: {(move_dists.abs() > 0.2).sum() / move_dists.shape[0] * 100}%")
    return ori_pts

def group_plane_via_normal(
    pts_normal, # S, 3
    pts_ins_assignment_masked, # S
    normal_angle_thresh=25,
    use_mean=True
):  
    normal_cos_thresh = math.cos(normal_angle_thresh/180.*np.pi)
    unique_labels = pts_ins_assignment_masked.unique()

    max_label_num = len(unique_labels)
    if unique_labels[0] == 0:
        max_label_num -= 1
    mean_normal = torch.zeros((max_label_num, 3), device="cuda")
    used_labels = []

    count = 0
    # for label in tqdm(unique_labels, total=len(unique_labels)):
    for label in unique_labels:
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)
        if use_mean:
            plane_normal = torch.mean(pts_normal[mask], dim=0)
        else:
            plane_normal = torch.median(pts_normal[mask], dim=0)[0]
        plane_normal = torch.mean(pts_normal[mask], dim=0)
        normal = plane_normal / (torch.norm(plane_normal) + 1e-10)
        mean_normal[count] = normal
        count += 1
    
    # calculate normal diff and dist diff
    normal_diff_nxn = (mean_normal[None, :] * mean_normal[:, None]).sum(2)

    # calculate normal and dist mask
    normal_mask_nxn = ((normal_diff_nxn > normal_cos_thresh))
    normal_mask_nxn = normal_mask_nxn | normal_mask_nxn.t()

    device = normal_diff_nxn.device

    final_mask = normal_mask_nxn
    final_mask.fill_diagonal_(False)

    all_idx = torch.arange(max_label_num).to(device)
    pts_ins_assignment_masked_tmp = pts_ins_assignment_masked.clone()

    for idx in range(max_label_num):
        label = used_labels[idx]
        if label == 0:
            continue
        mask = pts_ins_assignment_masked_tmp == label
        if mask.sum() == 0:
            continue
        inlier_idx = all_idx[final_mask[idx]]
        if len(inlier_idx) > 0:
            inlier_idx = inlier_idx.cpu().numpy().tolist()
            for cur_idx in inlier_idx:
                cur_label = used_labels[cur_idx]
                pts_ins_assignment_masked_tmp[pts_ins_assignment_masked_tmp==cur_label] = label

    return pts_ins_assignment_masked_tmp

def update_pts_normal(pts_plane_assignment_tensor, pts_normal_tensor, use_median=False):
    pts_normal_tensor = pts_normal_tensor.clone()
    #  ----------------------------------- update pts' normal
    max_label = pts_plane_assignment_tensor.max() + 1
    largest_ass = -1
    largest_size = 0
    for label in range(max_label):
        if label > 0:
            mask = pts_plane_assignment_tensor == label
            if mask.sum() == 0:
                continue
            if mask.sum() > largest_size:
                largest_size = mask.sum()
                largest_ass = label
            if use_median:
                normal_cur = torch.median(pts_normal_tensor[mask], dim=0)[0]
            else:
                normal_cur = torch.mean(pts_normal_tensor[mask], dim=0)
            try:
                pts_normal_tensor[mask] = normal_cur.reshape(1, 3).repeat(mask.sum(), 1)
            except:
                import pdb; pdb.set_trace()
    return pts_normal_tensor

def get_continues_pts_ins_assignment(pts_plane_assignment):
    labels = pts_plane_assignment.unique()
    pts_plane_assignment_new = torch.zeros_like(pts_plane_assignment)
    new_label = 1
    for label in labels:
        if label > 0:
            pts_plane_assignment_new[pts_plane_assignment == label] = new_label
            new_label += 1
    # assert pts_plane_assignment_new.min() > 0
    return pts_plane_assignment_new

# def calculate_pts2mesh_dist(pts, coarse_mesh_o3d):
#     mesh_vertices = torch.from_numpy(np.array(coarse_mesh_o3d.vertices)).to(pts.device).float()
    
#     pts_batch = pts.unsqueeze(0)  # (1, N, 3)
#     mesh_batch = mesh_vertices.unsqueeze(0)  # (1, M, 3)
    
#     dists, idx, _ = knn_points(pts_batch, mesh_batch, K=1)
    
#     pts_dist2mesh_tensor = torch.sqrt(dists.squeeze(-1).squeeze(0))
    
#     return pts_dist2mesh_tensor

def calculate_pts2mesh_dist(pts, coarse_mesh_o3d):
    mesh_vertices = torch.from_numpy(np.array(coarse_mesh_o3d.vertices)).to(pts.device).float()
    # pts: (N, 3) tensor on GPU
    # mesh_vertices: (M, 3) tensor on GPU
    
    x_i = LazyTensor(pts[:, None, :])      # (N, 1, 3)
    y_j = LazyTensor(mesh_vertices[None, :, :]) # (1, M, 3)
    
    # Symbolic distance computation (doesn't materialize the large matrix)
    dist_sq = ((x_i - y_j) ** 2).sum(-1)
    
    # Get the minimum squared distance for each point
    min_dist_sq = dist_sq.min(dim=1).squeeze()
    
    return torch.sqrt(min_dist_sq)

def sample_pts_from_GivenPlanePrim(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        space_resolution=0.05, 
        plane_ins_id=None
        ):
    """
    space_resolution=0.05 --> 0.05 meter
    """

    # What will happen if I have space_resolution=0.02 and a (0.01x0.01) plane??
    # 
    # With that exact setup, it is treated as a very tiny primitive and 'usually' gets dropped.
    # With nx, ny computed as:
    # nx, ny = int(max(2 * rx // sample_interval + 1, 3)), int(max(2 * ry // sample_interval + 1, 3))
    # Because of max(..., 3), even a tiny plane still gets at least:
    # nx = 3
    # ny = 3
    # total sampled points = 9
    # 
    # Then later, small instances are removed if `point count < min_pts_num`
    # And min_pts_num is forced to be at least:
    # max((0.1 / 0.02)^2, 25) = 25
    # So a standalone 0.01 x 0.01 plane (sampled as 9 points) fails 9 < 25 and gets removed.

    sample_interval = space_resolution

    rot_q = F.normalize(plane_rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    plane_normal_standard = torch.zeros_like(plane_normal)
    plane_normal_standard[..., -1] = 1
    radii_x = plane_radii[..., 0]  # n
    radii_y = plane_radii[..., 1]  # n

    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]) + 1 # valid plane ins IDs start from 1
    plane_ins_id = plane_ins_id.reshape(-1).int()  # 0 means non-plane ins
            
    pts = []
    per_point_plane_assignment = []
    normals = []
    faces = None

    for i in tqdm(range(len(plane_normal)), desc='sampleing points from planes...'):
        rx, ry = radii_x[i].item(), radii_y[i].item()
        nx, ny = int(max(2 * rx // sample_interval + 1, 3)), int(max(2 * ry // sample_interval + 1, 3))

        px = torch.linspace(-rx, rx, nx).reshape(-1, 1, 1).repeat(1, ny, 1).cuda()
        py = torch.linspace(-ry, ry, ny).reshape(1, -1, 1).repeat(nx, 1, 1).cuda()
        pz = torch.zeros_like(py)
        pts_i = torch.cat([px, py, pz], dim=-1).reshape(-1, 3)
        pts_i_transformed = torch.mm(rot_matrix[i], pts_i.permute(1, 0)).permute(1, 0) + plane_center[i].reshape(1, 3)

        normal_i = plane_normal[i].reshape(1, 3).repeat(pts_i_transformed.shape[0], 1)
        normals.append(normal_i)

        pts.append(pts_i_transformed)

        pts_planeID = plane_ins_id[i] * torch.ones(pts_i_transformed.shape[0]).to(pts_i_transformed.device)
        per_point_plane_assignment.append(pts_planeID.int())

        faces_i = []
        for i in range(nx*ny - ny):
            if (i+1) % (ny) == 0:
                continue
            faces_i.append([i, i+ny, i+1])
            faces_i.append([i+1, i+ny, i+ny+1])
        faces_i = torch.from_numpy(np.array(faces_i)).cuda()
        
        if faces is None:
            faces = faces_i
            cur_pts_num = nx*ny
        else:
            faces_i += cur_pts_num
            faces = torch.cat([faces, faces_i], dim=0)
            cur_pts_num += nx*ny

    pts = torch.cat(pts, dim=0).detach()

    faces = faces.detach()
    normals = torch.cat(normals, dim=0).detach()

    per_point_plane_assignment = torch.cat(per_point_plane_assignment, dim=0).detach()

    return pts, per_point_plane_assignment, normals, faces


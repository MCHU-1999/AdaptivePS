import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from random import randint
import math
from loguru import logger
import open3d as o3d
from .net_wrapper import PlanarRecWrapper

from utils.misc_util import setup_logging, get_train_param, save_config_files, prepare_folders, get_class
from utils.trainer_util import resume_model, calculate_plane_depth, plot_plane_img, save_checkpoints
from utils.mesh_util import get_coarse_mesh
# from utils.merge_util import merge_plane
from utils.merge_util_new import merge_plane
from utils.loss_util import normal_loss, metric_depth_loss
from utils.model_util import split_planes_xy_via_mask
import trimesh
import rerun as rr

class PlanarSplatTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.conf = kwargs['conf']
        self.expname, self.tag, self.timestamp, is_continue = get_train_param(kwargs, self.conf)
        # self.expname, self.timestamp, is_continue = get_train_param(kwargs, self.conf)
        self.expdir, self.plane_plots_dir, self.checkpoints_path, self.model_subdir = prepare_folders(kwargs, self.expname, self.timestamp)
        setup_logging(os.path.join(self.expdir, 'train.log'))
        kwargs['data']['expdir'] = self.expdir
        
        logger.info('Shell command : {0}'.format(' '.join(sys.argv)))
        save_config_files(self.expdir, self.conf)

        # =======================================  loading dataset
        logger.info('Loading data...')
        if 'data' in kwargs:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(kwargs['data'], **self.conf.get_config('dataset'))
        else:
            self.dataset = get_class(self.conf.get_string('train.dataset_class'))(**self.conf.get_config('dataset'))
        self.ds_len = self.dataset.n_images
        self.H = self.conf.dataset.img_res[0]
        self.W = self.conf.dataset.img_res[1]

        # Parse voxel_length and stuff from dataset config
        self.voxel_length = self.conf.get_float('dataset.voxel_length', default=0.02)
        self.sdf_trunc = self.conf.get_float('dataset.sdf_trunc', default=0.08)
        self.max_depth = self.conf.get_float('dataset.max_depth', default=20.0)
        self.depth_trunc = self.dataset.depth_trunc

        logger.info('Data loaded. Frame number = {0}'.format(self.ds_len))

        # =======================================  build plane model
        self.plane_model_conf = self.conf.get_config('plane_model')
        self.conf['dataset']['mesh_path'] = self.dataset.mono_mesh_dest

        net = PlanarRecWrapper(self.conf, self.plane_plots_dir)
        self.net = net.cuda()
        self.resumed = False
        self.start_iter = resume_model(self) if is_continue else 0
        self.iter_step = self.start_iter
        self.net.build_optimizer_and_LRscheduler()

        # ======================================= plot settings
        self.do_vis = kwargs['do_vis']
        self.plot_freq = self.conf.get_int('train.plot_freq')        
        
        # ======================================= loss settings
        loss_plane_conf = self.conf.get_config('plane_model.plane_loss')
        self.weight_plane_normal = loss_plane_conf.get_float('weight_mono_normal')
        self.weight_plane_depth = loss_plane_conf.get_float('weight_mono_depth')

        # ======================================= training settings
        self.max_total_iters = self.conf.get_int('train.max_total_iters')
        self.process_plane_freq_ite = self.conf.get_int('train.process_plane_freq_ite')
        self.coarse_stage_ite = self.conf.get_int('train.coarse_stage_ite')
        self.split_start_ite = self.conf.get_int('train.split_start_ite')
        self.check_vis_freq_ite = self.conf.get_int('train.check_plane_vis_freq_ite')
        self.data_order = self.conf.get_string('train.data_order')

    def run(self):
        self.train()
        self.merger()
        logger.info(f'\n====================\nFinished\n====================\n\n')
    
    def train(self):
        logger.info("Training...")
        if self.start_iter >= self.max_total_iters:
            return
        weight_decay_list = []
        for i in tqdm(range(self.max_total_iters+1), desc="generating sampling idx list..."):
            weight_decay_list.append(max(math.exp(-i / self.max_total_iters), 0.1))
        logger.info('Start training at {:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
        self.net.train()
        if self.iter_step == 0:
            # For the very first one we really don't care that much, right?
            self.check_plane_visibility_cuda()

        view_info_list = None
        progress_bar = tqdm(range(self.start_iter, self.max_total_iters+1), desc="Training progress")
        calculate_plane_depth(self)
        for iter in range(self.start_iter, self.max_total_iters + 1):
            self.iter_step = iter
            # ======================================= process planes
            if iter > self.coarse_stage_ite and iter % self.process_plane_freq_ite==0:  
                self.net.regularize_plane_shape()
                self.net.prune_small_plane()
                if iter > self.split_start_ite and iter <= self.max_total_iters - 1000:
                    logger.info('splitting...')
                    ori_num = self.net.planarSplat.get_plane_num()
                    self.net.split_plane()
                    new_num = self.net.planarSplat.get_plane_num()
                    logger.info(f'plane num: {ori_num} ---> {new_num}')
            # ======================================= get view info
            if not view_info_list:
                view_info_list = self.dataset.view_info_list.copy()
            if self.data_order == 'rand':
                view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            else:
                view_info = view_info_list.pop(0)
            raster_cam_w2c = view_info.raster_cam_w2c
            # ======================================= zero grad
            self.net.optimizer.zero_grad()
            #  ======================================= plane forward
            allmap = self.net.planarSplat(view_info,iter)
            # ------------ get rendered maps
            depth = allmap[0:1].squeeze().view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # ------------ get aux maps
            vis_weight = allmap[1:2].squeeze().view(-1)
            valid_ray_mask = vis_weight > 0.00001
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_depth_mask = view_info.mono_depth.abs() > 0
            valid_ray_mask = valid_ray_mask & valid_depth_mask & valid_normal_mask

            # ======================================= calculate losses
            loss_final = 0.
            decay = weight_decay_list[iter]
            # ------------ calculate plane loss
            loss_plane_normal_l1, loss_plane_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_plane_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask, max_depth=self.max_depth)
            loss_plane = (loss_plane_depth * 1.0) * self.weight_plane_depth \
                        + (loss_plane_normal_l1 + loss_plane_normal_cos) * self.weight_plane_normal
            loss_final += loss_plane * decay

            # ======================================= backward & update plane denom & update learning rate
            loss_final.backward()
            self.net.optimizer.step()
            self.net.update_grad_stats()
            self.net.regularize_plane_shape(empty_cache=False)
            image_index = view_info.index
            self.dataset.view_info_list[image_index].plane_depth = depth.detach().clone()

            with torch.no_grad():
                # Progress bar
                plane_num = self.net.planarSplat.get_plane_num()
                if iter % 100 == 0:
                    loss_dict = {
                        "Planes": f"{plane_num}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(100)
                if iter == self.max_total_iters:
                    progress_bar.close()
            
            # ======================================= plot model outputs
            if self.do_vis and iter % self.plot_freq == 0:  # do_vis is often False
                self.net.regularize_plane_shape()
                self.net.eval()
                self.net.planarSplat.draw_plane(epoch=iter)
                plot_plane_img(self)
                self.net.train()
            
            if iter > self.coarse_stage_ite and iter % self.check_vis_freq_ite == 0:
                self.check_plane_visibility_cuda_plus_plus()
            elif iter > 0 and iter % self.check_vis_freq_ite == 0:
                self.check_plane_visibility_cuda()
        
        self.check_plane_visibility_cuda_plus_plus(lastdog=True)
        save_checkpoints(self, iter=self.iter_step, only_latest=False)

    def merger(self, save_mesh=True, debug_output=True):
        logger.info("Merging 3D planar primitives...")
        output_dir = self.conf.get_string('train.rec_folder_name', default='')
        if len(output_dir) == 0:
            output_dir = self.expdir
        self.net.eval()
        save_root = output_dir
        # os.makedirs(save_root, exist_ok=True)

        ## prune planes whose maximum radii lower than the threshold
        self.net.prune_small_plane(min_radii=0.02)
        logger.info("number of 3D planar primitives = %d"%(self.net.planarSplat.get_plane_num()))

        # # The 3rd version, but using mesh to trim is kinda unfair
        # ref_mesh = trimesh.load_mesh(self.dataset.mono_mesh_dest)
        # -----------------------------------------------------------
        # The 2nd version
        ref_mesh = get_coarse_mesh(
            self.net, 
            self.dataset.view_info_list.copy(), 
            self.H, 
            self.W, 
            voxel_length=self.voxel_length, 
            sdf_trunc=self.sdf_trunc,
            depth_trunc=self.depth_trunc
        )
        if debug_output:
            save_path = os.path.join(save_root, f"ref_mesh.ply")
            logger.info(f'saving reference mesh to {save_path} ---DEBUG')
            o3d.io.write_triangle_mesh(
                        save_path, 
                        ref_mesh)
        # -----------------------------------------------------------
        # # The very original verison looks like this:
        # ref_mesh = get_coarse_mesh(
        #     self.net, 
        #     self.dataset.view_info_list.copy(), 
        #     self.H, 
        #     self.W, 
        #     voxel_length=0.02, 
        #     sdf_trunc=0.08
        # )
        
        merge_config_coarse = self.conf.get_config('merge_coarse', default=None)
        merge_config_fine = self.conf.get_config('merge_fine', default=None)
        if merge_config_coarse is not None:
            logger.info(f'mergeing (coarse)...')
            planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                self.net, 
                ref_mesh, 
                plane_ins_id=None,
                # New parameters for trimming bg points
                view_info_list=self.dataset.view_info_list,
                H=self.H, W=self.W,
                depth_trunc=self.depth_trunc,
                # The rest
                **merge_config_coarse
            )
            if debug_output:
                save_path = os.path.join(save_root, f"coarse_planar_mesh.ply")
                logger.info(f'saving 1st merged (coarse) mesh to {save_path} ---DEBUG')
                o3d.io.write_triangle_mesh(
                            save_path, 
                            planarSplat_eval_mesh)
                
            if merge_config_fine is not None:
                logger.info(f'mergeing (fine)...')
                planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                    self.net, 
                    ref_mesh, 
                    plane_ins_id=plane_ins_id_new,
                    # New parameters for trimming bg points
                    view_info_list=self.dataset.view_info_list,
                    H=self.H, W=self.W,
                    depth_trunc=self.depth_trunc,
                    # The rest
                    **merge_config_fine
                )
        else:
            raise ValueError("No merge configuration found!")
        
        if save_mesh:
            # Original code: using o3d.io.write_triangle_mesh()
            save_path = os.path.join(save_root, f"final_planar_mesh.ply")
            logger.info(f'saving final planar mesh to {save_path}')
            o3d.io.write_triangle_mesh(
                        save_path, 
                        planarSplat_eval_mesh)

            # # The other way: using trimesh (buggy)
            # save_path = os.path.join(save_root, f"final_planar_mesh.ply")
            # logger.info(f'saving final planar mesh to {save_path}')
            # planarSplat_eval_mesh.export(save_path)
        
        return planarSplat_eval_mesh

    def check_plane_visibility_cuda(self):   
        self.net.regularize_plane_shape(empty_cache=False)     
        logger.info('checking plane visibility...')
        self.net.eval()
        self.net.reset_plane_vis()
        view_info_list = self.dataset.view_info_list.copy()
        for iter in tqdm(range(self.ds_len)):
            # ========================= get view info
            view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            raster_cam_w2c = view_info.raster_cam_w2c
            # ----------- plane forward
            allmap = self.net.planarSplat(view_info, self.iter_step)
            # get rendered maps
            depth = allmap[0:1].view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # get aux maps
            vis_weight = allmap[1:2].view(-1)
            valid_ray_mask = vis_weight > 0.00001

            loss_final = 0.
            # ======================================= calculate plane losses
            loss_mono_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask, max_depth=self.max_depth)
            loss_normal_l1, loss_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_final += loss_mono_depth + loss_normal_cos + loss_normal_l1

            loss_final.backward() 
            # update plane visibility
            self.net.update_plane_vis() 
            self.net.optimizer.zero_grad()

        self.net.optimizer.zero_grad()
        self.net.train()
        self.net.prune_invisible_plane()
        self.net.planarSplat.draw_plane(epoch=self.iter_step)
        
    def check_plane_visibility_cuda_plus_plus(self, debug=True, lastdog=False):
        """
        One single plane can be contributing losses in both FG and BG, if we only look at FG/BG individually then
        we're not really controlling the behavior as we wanted. Here's what we do:

        FG only: normal training.
        BG only: prune.
        FG + BG: split.
        Neither: prune.
        """
        EPSILON = 1e-6
        FG_THRESHOLDING = 0.6
        COUNTERPART = 0.3
        MAX_SPLIT_PER_CHECK = 1500

        self.net.regularize_plane_shape(empty_cache=False)
        logger.info('checking plane visibility (FG/BG split)...')
        self.net.eval()

        plane_num = self.net.planarSplat.get_plane_num()
        fg_hit_count = torch.zeros((plane_num, 1), device='cuda')
        bg_hit_count = torch.zeros((plane_num, 1), device='cuda')

        view_info_list = self.dataset.view_info_list.copy()
        for _ in tqdm(range(self.ds_len)):
            view_info = view_info_list.pop(randint(0, len(view_info_list) - 1))
            raster_cam_w2c = view_info.raster_cam_w2c

            allmap = self.net.planarSplat(view_info, self.iter_step)
            depth = allmap[0:1].view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1, 2, 0) @ (raster_cam_w2c[:3, :3].T)).view(-1, 3)
            vis_weight = allmap[1:2].view(-1)

            valid_ray_mask = vis_weight > 1e-5
            fg_area = view_info.fg_mask
            fg_mask = valid_ray_mask & fg_area
            bg_mask = valid_ray_mask & ~fg_area

            # FG pass
            self.net.optimizer.zero_grad()
            fg_vis = torch.zeros(plane_num, device='cuda', dtype=torch.bool)
            if fg_mask.sum() > 0:
                loss_mono_depth = metric_depth_loss(depth, view_info.mono_depth, fg_mask, max_depth=self.max_depth)
                loss_normal_l1, loss_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, fg_mask)
                fg_loss = loss_mono_depth + loss_normal_cos + loss_normal_l1
                if torch.isfinite(fg_loss):
                    fg_loss.backward(retain_graph=True)
                    fg_vis = self.net.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
                    fg_hit_count[fg_vis] += 1
            self.net.optimizer.zero_grad()

            # BG pass
            bg_vis = torch.zeros(plane_num, device='cuda', dtype=torch.bool)
            if bg_mask.sum() > 0:
                bg_loss = vis_weight[bg_mask].mean()
                if torch.isfinite(bg_loss):
                    bg_loss.backward()
                    bg_vis = self.net.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
                    bg_hit_count[bg_vis] += 1
            self.net.optimizer.zero_grad()

        fg_hits = fg_hit_count.squeeze(-1)
        bg_hits = bg_hit_count.squeeze(-1)

        fg_ratio = fg_hits / (fg_hits + bg_hits + EPSILON)
        fg_only = (fg_ratio > FG_THRESHOLDING)
        bg_only = (fg_ratio <= COUNTERPART)
        ambiguous = (fg_ratio > COUNTERPART) & (fg_ratio <= FG_THRESHOLDING)

        # DEBUG -------------------------------------------------------------------------------
        if debug:
            category_id = torch.full((plane_num,), 3, device='cuda', dtype=torch.long)
            category_id[fg_only] = 0
            category_id[bg_only] = 1
            category_id[ambiguous] = 2

            self.net.planarSplat.draw_plane_debug(
                epoch=self.iter_step,
                plane_id=category_id,
                save_mesh=True,
            )
            logger.info('[check_plane_visibility++] saved debug mesh')
        # -------------------------------------------------------------------------------------

        keep_mask = ~bg_only

        if bg_only.any():
            self.net.prune_core(bg_only.detach())
            logger.info(f'[check_plane_visibility++] removed {int(bg_only.sum().item())} planes')

        if (not lastdog) and ambiguous.any():
            ambiguous_count = ambiguous.sum().item()
            
            if ambiguous_count > MAX_SPLIT_PER_CHECK:
                # Option 1: Sort by ambiguity (lowest first) and take top-k lowest
                ambiguity_score = (1 - fg_ratio[ambiguous])
                _, topk = torch.topk(ambiguity_score, MAX_SPLIT_PER_CHECK, largest=False)
                
                split_mask = torch.zeros(plane_num, dtype=torch.bool, device='cuda')
                ambiguous_idx = ambiguous.nonzero(as_tuple=False).squeeze(-1)
                split_mask[ambiguous_idx[topk]] = True
            else:
                split_mask = ambiguous
            
            # prune_core above changes plane indexing; remap split candidates into post-prune index space
            split_mask = split_mask[keep_mask]

            if split_mask.any():
                self.net.split_selected_planes(split_mask)
                logger.info(f'[check_plane_visibility++] split {int(split_mask.sum().item())} planes (capped at {MAX_SPLIT_PER_CHECK})')
            else:
                logger.info(f'[check_plane_visibility++] no valid ambiguous planes to split after pruning')
        else:
            logger.info(f'[check_plane_visibility++] LASTDOG no split')

        self.net.planarSplat.check_model()
        logger.info(
            f'[check_plane_visibility++] fg_only={int(fg_only.sum().item())}, '
            f'bg_only={int(bg_only.sum().item())}, ambiguous={int(ambiguous.sum().item())}'
        )

        self.net.optimizer.zero_grad()
        self.net.train()
        self.net.planarSplat.draw_plane(epoch=self.iter_step)

    # def check_plane_visibility_cuda_plus_plus(self, debug=True, lastdog=False):
    #     """
    #     Percentile-based plane split/prune policy using contribution scores.
        
    #     Strategy:
    #     - Compute quality_score = fg_hits - LAMBDA * bg_hits (contribution metric)
    #     - Prune: worst 10% by quality_score
    #     - Split: middle 50% (10th to 60th percentile)
    #     - Keep: best 40% (60th percentile and above)
        
    #     This provides adaptive, non-manual-threshold-based decisions.
    #     """
    #     LAMBDA = 1.0  # Balance FG vs BG contributions
    #     PRUNE_PERCENTILE = 10
    #     SPLIT_PERCENTILE = 60
    #     MAX_SPLIT_PER_CHECK = 1500

    #     self.net.regularize_plane_shape(empty_cache=False)
    #     logger.info('checking plane visibility (FG/BG split)...')
    #     self.net.eval()

    #     plane_num = self.net.planarSplat.get_plane_num()
    #     fg_hit_count = torch.zeros((plane_num, 1), device='cuda')
    #     bg_hit_count = torch.zeros((plane_num, 1), device='cuda')

    #     view_info_list = self.dataset.view_info_list.copy()
    #     for _ in tqdm(range(self.ds_len)):
    #         view_info = view_info_list.pop(randint(0, len(view_info_list) - 1))
    #         raster_cam_w2c = view_info.raster_cam_w2c

    #         allmap = self.net.planarSplat(view_info, self.iter_step)
    #         depth = allmap[0:1].view(-1)
    #         normal_local_ = allmap[2:5]
    #         normal_global = (normal_local_.permute(1, 2, 0) @ (raster_cam_w2c[:3, :3].T)).view(-1, 3)
    #         vis_weight = allmap[1:2].view(-1)

    #         valid_ray_mask = vis_weight > 1e-5
    #         fg_area = view_info.fg_mask
    #         fg_mask = valid_ray_mask & fg_area
    #         bg_mask = valid_ray_mask & ~fg_area

    #         # FG pass
    #         self.net.optimizer.zero_grad()
    #         fg_vis = torch.zeros(plane_num, device='cuda', dtype=torch.bool)
    #         if fg_mask.sum() > 0:
    #             loss_mono_depth = metric_depth_loss(depth, view_info.mono_depth, fg_mask, max_depth=self.max_depth)
    #             loss_normal_l1, loss_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, fg_mask)
    #             fg_loss = loss_mono_depth + loss_normal_cos + loss_normal_l1
    #             if torch.isfinite(fg_loss):
    #                 fg_loss.backward(retain_graph=True)
    #                 fg_vis = self.net.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
    #                 fg_hit_count[fg_vis] += 1
    #         self.net.optimizer.zero_grad()

    #         # BG pass
    #         bg_vis = torch.zeros(plane_num, device='cuda', dtype=torch.bool)
    #         if bg_mask.sum() > 0:
    #             bg_loss = vis_weight[bg_mask].mean()
    #             if torch.isfinite(bg_loss):
    #                 bg_loss.backward()
    #                 bg_vis = self.net.planarSplat._plane_center.grad.abs().detach().sum(dim=-1) > 0
    #                 bg_hit_count[bg_vis] += 1
    #         self.net.optimizer.zero_grad()

    #     fg_hits = fg_hit_count.squeeze(-1)
    #     bg_hits = bg_hit_count.squeeze(-1)

    #     # Compute quality score: FG contribution minus weighted BG contribution
    #     quality_score = fg_hits.float() - LAMBDA * bg_hits.float()

    #     # Determine thresholds using percentiles
    #     prune_threshold = torch.quantile(quality_score, PRUNE_PERCENTILE / 100.0)
    #     split_threshold = torch.quantile(quality_score, SPLIT_PERCENTILE / 100.0)

    #     # Classify planes by quality percentile
    #     prune_mask = (quality_score < prune_threshold)
    #     split_mask = (quality_score >= prune_threshold) & (quality_score < split_threshold)

    #     # Apply split budget cap: keep only top scorers within split category
    #     split_count = split_mask.sum().item()
    #     if split_count > MAX_SPLIT_PER_CHECK:
    #         split_indices = split_mask.nonzero(as_tuple=False).squeeze(-1)
    #         split_scores = quality_score[split_indices]
    #         _, topk_idx = torch.topk(split_scores, MAX_SPLIT_PER_CHECK, largest=True)
            
    #         split_mask_new = torch.zeros_like(split_mask)
    #         split_mask_new[split_indices[topk_idx]] = True
    #         split_mask = split_mask_new

    #     # DEBUG -------------------------------------------------------------------------------
    #     if debug:
    #         keep_mask_classification = (quality_score >= split_threshold)
    #         category_id = torch.full((plane_num,), 3, device='cuda', dtype=torch.long)
    #         category_id[keep_mask_classification] = 0  # Good (top 40%)
    #         category_id[split_mask] = 1                 # Ambiguous (middle 50%)
    #         category_id[prune_mask] = 2                 # Bad (bottom 10%)

    #         self.net.planarSplat.draw_plane_debug(
    #             epoch=self.iter_step,
    #             plane_id=category_id,
    #             save_mesh=True,
    #         )
    #         logger.info('[check_plane_visibility++] saved category-colored lastdog mesh')
    #     # -------------------------------------------------------------------------------------

    #     keep_mask = ~prune_mask

    #     if prune_mask.any():
    #         self.net.prune_core(prune_mask.detach())
    #         logger.info(f'[check_plane_visibility++] pruned {int(prune_mask.sum().item())} planes (worst {PRUNE_PERCENTILE}%)')

    #     # Remap split_mask to post-prune index space
    #     split_mask = split_mask[keep_mask]

    #     if (not lastdog) and split_mask.any():
    #         self.net.split_selected_planes(split_mask)
    #         logger.info(f'[check_plane_visibility++] split {int(split_mask.sum().item())} planes (percentiles {PRUNE_PERCENTILE}-{SPLIT_PERCENTILE}%, capped at {MAX_SPLIT_PER_CHECK})')
    #     else:
    #         if lastdog:
    #             logger.info(f'[check_plane_visibility++] LASTDOG no split')
    #         else:
    #             logger.info(f'[check_plane_visibility++] no planes selected for split')

    #     self.net.planarSplat.check_model()
    #     logger.info(
    #         f'[check_plane_visibility++] score_range=[{quality_score.min():.3f}, {quality_score.max():.3f}], '
    #         f'prune_th={prune_threshold:.3f}, split_th={split_threshold:.3f}, '
    #         f'pruned={int(prune_mask.sum().item())}, split={int(split_mask.sum().item())}'
    #     )

    #     self.net.optimizer.zero_grad()
    #     self.net.train()
    #     self.net.planarSplat.draw_plane(epoch=self.iter_step)
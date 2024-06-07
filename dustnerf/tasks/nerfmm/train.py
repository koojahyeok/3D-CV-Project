import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_colmap import DataLoaderWithCOLMAP
from dataloader.with_blender import DataLoaderWithBLENDER
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.comp_ate import compute_ate
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--eval_img_interval', default=100, type=int, help='eval images every this epoch number')
    parser.add_argument('--eval_cam_interval', default=5, type=int, help='eval camera params every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='/home/jahyeok/share_data/nerf_data',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='LLFF/fern')
    parser.add_argument('--use_ndc', default=True, type=eval, choices=[True, False])

    parser.add_argument('--nerf_lr', default=0.001, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=eval, choices=[True, False])
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--focal_lr', default=0.005, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--store_pose_history', type=bool, default=True, help='store pose history to log dir')

    parser.add_argument('--start_refine_pose_epoch', type=int, default=-1,
                        help='Set to -1 to init pose from identity matrices. Set to a epoch number >= 0 '
                             'to init poses from COLMAP and start refining them from this epoch.')
    parser.add_argument('--start_refine_focal_epoch', type=int, default=-1,
                        help='Set to -1 to init focal from image resolution. Set to a epoch number >= 0 '
                             'to init focals from COLMAP and start refining them from this epoch.')

    parser.add_argument('--resize_ratio', type=int, default=1, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')
    parser.add_argument('--eval_img_num', type=int, default=1, help='num of images to eval')
    parser.add_argument('--eval_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--alias', type=str, default='', help="experiments alias")

    # which i added
    parser.add_argument('--dtype', type=str, default='BLENDER', help="COLMAP vs BLENDER data format")
    parser.add_argument('--model', type=str, default='nerfmm', help='choose nerf model')
    
    # arguments for LearnFocal model initialization
    parser.add_argument('--init_weight', type=float, default=3, help='initial value of weight (LearnFocal)')
    parser.add_argument('--init_global_bias', type=float, default=-700, help='initial value of global bias (LearnFocal)')
    parser.add_argument('--dustparam_path_name', type=str, default='dust_params')
    
    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_Ntr_img_'+ str(args.train_img_num) + \
             '_freq_' + str(args.pos_enc_levels) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


def store_current_pose(pose_net, pose_history_dir, epoch_i):
    pose_net.eval()

    num_cams = pose_net.module.num_cams if isinstance(pose_net, torch.nn.DataParallel) else pose_net.num_cams

    c2w_list = []
    for i in range(num_cams):
        c2w = pose_net(i)  # (4, 4)
        c2w_list.append(c2w)

    c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
    c2w_list = c2w_list.detach().cpu().numpy()

    np.save(os.path.join(pose_history_dir, str(epoch_i).zfill(6)), c2w_list)
    return


def model_render_image(c2w, rays_cam, t_vals, near, far, H, W, fxfy, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :rgb_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
                                                                     H, W, fxfy, perturb_t)

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, 27)
    else:
        dir_enc = None

    # inference rgb and density using position and direction encoding.
    rgb_density = model(pos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'sample_pos': sample_pos,  # (H, W, N_sample, 3)
        'depth_map': depth_map,  # (H, W)
        'rgb_density': rgb_density,  # (H, W, N_sample, 4)
    }

    return result


def eval_one_epoch_img_nerfmm(eval_c2ws, scene_train, model, focal_net, pose_param_net,
                       my_devices, args, epoch_i, writer, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net.eval_forward()
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = eval_c2ws.shape[0]

    rendered_img_list = []
    rendered_depth_list = []

    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1))  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    writer.add_image('eval_depth', disp_depth, global_step=epoch_i)

def eval_one_epoch_img_nerf(eval_c2ws, scene_train, model, 
                       my_devices, args, epoch_i, writer, rgb_act_fn):
    model.eval()

    fxfy = scene_train.focal
    ray_dir_cam = scene_train.ray_dir_cam.to(my_devices)
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = eval_c2ws.shape[0]

    rendered_img_list = []
    rendered_depth_list = []

    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1))  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    writer.add_image('eval_depth', disp_depth, global_step=epoch_i)


def eval_one_epoch_traj(scene_train, pose_param_net):
    pose_param_net.eval()

    est_c2ws_train = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
    colmap_c2ws_train = scene_train.c2ws  # (N, 4, 4) torch
    stats_tran, stats_rot, stats_scale = compute_ate(est_c2ws_train, colmap_c2ws_train, align_a2b='sim3')

    return stats_tran, stats_rot, stats_scale


def train_one_epoch_nerfmm(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn, epoch_i):
    model.train()

    if epoch_i >= args.start_refine_pose_epoch:
        pose_param_net.train()
    else:
        pose_param_net.eval()

    if epoch_i >= args.start_refine_focal_epoch:
        focal_net.train()
    else:
        focal_net.eval()

    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    # shuffle the training imgs
    ids = np.arange(N_img)
    np.random.shuffle(ids)

    for i in ids:
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)

        if epoch_i >= args.start_refine_focal_epoch:
            fxfy = focal_net(i)
            ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        else:
            raise ValueError('')
            fxfy = scene_train.focal[i]
            ray_dir_cam = scene_train.ray_dir_cam[i].to(my_devices)

        if epoch_i >= args.start_refine_pose_epoch:
            c2w = pose_param_net(i)  # (4, 4)
        else:
            with torch.no_grad():
                c2w = pose_param_net(i)  # (4, 4)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses

def train_one_epoch_nerf(scene_train, optimizer_nerf, model,
                    my_devices, args, rgb_act_fn, epoch_i):
    model.train()

    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    # shuffle the training imgs
    ids = np.arange(N_img)
    np.random.shuffle(ids)

    for i in ids:
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)

        fxfy = scene_train.focal
        ray_dir_cam = scene_train.ray_dir_cam.to(my_devices)
        c2w = scene_train.c2ws[i].to(my_devices)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_nerf.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/nerfmm', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/nerfmm/train.py', experiment_dir)

    if args.store_pose_history:
        pose_history_dir = Path(os.path.join(experiment_dir, 'pose_history'))
        pose_history_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(experiment_dir))

    '''Data Loading'''
    if args.dtype ==  'COLMAP':
        scene_train = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                        scene_name=args.scene_name,
                                        data_type='train',
                                        res_ratio=args.resize_ratio,
                                        num_img_to_load=args.train_img_num,
                                        skip=args.train_skip,
                                        use_ndc=args.use_ndc)
    else:
        scene_train = DataLoaderWithBLENDER(base_dir=args.base_dir,
                                        scene_name=args.scene_name,
                                        data_type='train',
                                        res_ratio=args.resize_ratio,
                                        num_img_to_load=args.train_img_num,
                                        skip=args.train_skip,
                                        use_ndc=args.use_ndc,
                                        camera_param_name=args.dustparam_path_name)
        

    # The COLMAP eval poses are not in the same camera space that we learned so we can only check NVS
    # with a 4x4 identity pose.
    eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    # learn focal parameter
    if args.model == 'nerfmm':
        print(scene_train.N_imgs)
        if args.start_refine_focal_epoch > -1:
            focal_net = LearnFocal(H=scene_train.H, 
                                   W=scene_train.W, 
                                   n_images=scene_train.N_imgs, 
                                   init_focal=scene_train.focal,
                                   init_weight=args.init_weight,
                                   init_global_bias=args.init_global_bias,
                                   order=args.focal_order,
                                   req_grad=args.learn_focal)
        else:
            focal_net = LearnFocal(H=scene_train.H, 
                                   W=scene_train.W, 
                                   n_images=scene_train.N_imgs, 
                                   init_focal=scene_train.focal,
                                   init_weight=args.init_weight,
                                   init_global_bias=args.init_global_bias,
                                   order=args.focal_order,
                                   req_grad=args.learn_focal)

        if args.multi_gpu:
            focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
        else:
            focal_net = focal_net.to(device=my_devices)

        # learn pose for each image
        # if scene_train.c2ws:
        if True:
            pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
        else:
            raise ValueError("Initial value must be provided.")
            pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
        if args.multi_gpu:
            pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
        else:
            pose_param_net = pose_param_net.to(device=my_devices)


    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)

    if args.model == 'nerfmm':
        optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
        optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
                                                          gamma=args.nerf_lr_gamma)
    if args.model == 'nerfmm':
        scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
                                                            gamma=args.focal_lr_gamma)
        scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
                                                            gamma=args.pose_lr_gamma)

    '''Training'''
    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        if args.model == 'nerfmm':
            train_epoch_losses = train_one_epoch_nerfmm(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose,
                                                model, focal_net, pose_param_net, my_devices, args, rgb_act_fn, epoch_i)
        elif args.model == 'nerf':
            train_epoch_losses = train_one_epoch_nerf(scene_train, optimizer_nerf, 
                                                model, my_devices, args, rgb_act_fn, epoch_i)
        
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()
        if args.model == 'nerfmm':
            scheduler_focal.step()
            scheduler_pose.step()

        train_psnr = mse2psnr(train_L2_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        pose_history_milestone = list(range(0, 100, 5)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
        if epoch_i in pose_history_milestone:
            with torch.no_grad():
                if args.store_pose_history and args.model=='nerfmm':
                    store_current_pose(pose_param_net, pose_history_dir, epoch_i)

        if args.model == 'nerfmm':
            if epoch_i % args.eval_cam_interval == 0 and epoch_i > 0:
                with torch.no_grad():
                    eval_stats_tran, eval_stats_rot, eval_stats_scale = eval_one_epoch_traj(scene_train, pose_param_net)
                    writer.add_scalar('eval/traj/translation', eval_stats_tran['mean'], epoch_i)
                    writer.add_scalar('eval/traj/rotation', eval_stats_rot['mean'], epoch_i)
                    writer.add_scalar('eval/traj/scale', eval_stats_scale['mean'], epoch_i)

                    logger.info('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
                                                                                                                        eval_stats_tran['mean'],
                                                                                                                        eval_stats_rot['mean'],
                                                                                                                        eval_stats_scale['mean']))
                    tqdm.write('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
                                                                                                                        eval_stats_tran['mean'],
                                                                                                                        eval_stats_rot['mean'],
                                                                                                                        eval_stats_scale['mean']))

                    fxfy = focal_net.get_all_focals()
                    tqdm.write(f'Est focals: {fxfy}')
                    tqdm.write(f'focal coeff: W {focal_net.weight_global} bg {focal_net.bias_global} bl {focal_net.bias_local}')
                    # logger.info('Est fx: {0:.2f}, fy {1:.2f}, COLMAP focal: {2:.2f}'.format(fxfy[0].item(), fxfy[1].item(),
                    #                                                                         scene_train.focal.mean()))
                    if torch.is_tensor(fxfy):
                        L1_focal = torch.abs(fxfy.cpu() - scene_train.focal.unsqueeze(0).cpu()).mean().item()
                    else:
                        L1_focal = np.abs(fxfy - scene_train.focal).mean()
                    writer.add_scalar('eval/L1_focal', L1_focal, epoch_i)

        if epoch_i % args.eval_img_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                if args.model == 'nerfmm':
                    eval_one_epoch_img_nerfmm(eval_c2ws, scene_train, model, focal_net, pose_param_net, my_devices,
                                       args, epoch_i, writer, rgb_act_fn)
                elif args.model == 'nerf':
                    eval_one_epoch_img_nerf(eval_c2ws, scene_train, model, my_devices,
                                       args, epoch_i, writer, rgb_act_fn)

                # save the latest model.
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                if args.model == 'nerfmm':
                    save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                    save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)

# todo
# dataset multi focal length
# rotation
# debugging

# python tasks/nerfmm/spiral.py --base_dir='/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/3D-CV-Project/data/' --scene_name='lego' --dtype='BLENDER' --ckpt_dir='/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/3D-CV-Project/dustnerf/logs/nerfmm/lego/lr_0.001_gpu0_seed_17_resize_1_Nsam_128_Ntr_img_-1_freq_10__240607_2013'
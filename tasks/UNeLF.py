import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging
import imageio
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import lpips as lpips_lib
sys.path.append(os.path.join(sys.path[0], '../'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, load_ckpt_to_net, save_checkpoint
from utils.volume_op import volume_sampling_ndc
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.intrinsics import LearnFocal
from models.poses import LearnPose
from models.model import UNeLF
from models.scale import scale
from third_party import pytorch_ssim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--eval_interval', default=1000, type=int, help='run eval every this epoch number')

    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='/home/zmybuaa/UNeLF')
    parser.add_argument('--scene_name', type=str, default='boxes')

    parser.add_argument('--nerf_lr', default=0.001, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--st_scale', type=float, default=0.125)
    parser.add_argument('--st_lr', default=0.001, type=float)
    parser.add_argument('--st_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--st_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--resize_ratio', type=int, default=1, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')
    parser.add_argument('--rows', type=int, default=5)

    return parser.parse_args()

def normalize(x):
    return x / torch.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = torch.cat([viewmatrix(vec2, up, center), hwf], 1)
    bottom = torch.tensor([0, 0, 0, 1.0]).view(1, 4).to(device=poses.device)
    c2w = torch.cat([c2w[:3,:4], bottom], -2)
    return c2w

def gen_detail_name(args):
    outstr = 'gpu_' + str(args.gpu_id) + \
             'st_' + str(args.st_scale) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr

def model_render_image(c2w, rays_cam, near, far, H, W, fxfy, st, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn):
    data_uvst = volume_sampling_ndc(c2w, rays_cam, near, far, H, W, fxfy, perturb_t, st)
    rgb = model(data_uvst)  
    H, W = rays_cam.shape[0], rays_cam.shape[1]
    rgb = rgb.reshape(H, W, 3)
    result = {
        'rgb': rgb,  # (H, W, 3)
    }
    return result

def eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net, st_net, 
                   my_devices, args, epoch_i, writer, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()
    st_net.eval()

    fxfy = focal_net(0)
    st = st_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    N_img, H, W = eval_c2ws.shape[0], scene_train.H, scene_train.W
    rendered_img_list = []

    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices)  
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy, st, 
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']
            rendered_img.append(rgb_rendered_rows)

        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_img_list.append(rendered_img.cpu().numpy())

    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1)) 
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    return

def eval_all(scene_eval, eval_c2ws, model, focal_net, pose_param_net, st_net, 
                   my_devices, args, rgb_act_fn, img_out_dir):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()
    st_net.eval()
    lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(my_devices)

    fxfy = focal_net(0)
    st = st_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_eval.H, scene_eval.W, fxfy[0], fxfy[1])
    N_img, H, W = eval_c2ws.shape[0], scene_eval.H, scene_eval.W

    rendered_img_list = []

    eval_mse_list = []
    eval_psnr_list = []
    eval_ssim_list = []
    eval_lpips_list = []
    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices) 
        gt = scene_eval.imgs[i].to(my_devices)
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, scene_eval.near, scene_eval.far,
                                               scene_eval.H, scene_eval.W, fxfy, st, 
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']  
            rendered_img.append(rgb_rendered_rows)

        rendered_img = torch.cat(rendered_img, dim=0)

        rendered_img_list.append(rendered_img.cpu().numpy())
        img = (rendered_img.cpu().numpy()* 255).astype(np.uint8)
        imageio.imwrite(os.path.join(img_out_dir, str(i).zfill(4) + '.png'), img)

        mse = F.mse_loss(rendered_img, gt).item()
        psnr = mse2psnr(mse)
        ssim = pytorch_ssim.ssim(rendered_img.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0)).item()
        lpips_loss = lpips_vgg_fn(rendered_img.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                  gt.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()

        eval_mse_list.append(mse)
        eval_psnr_list.append(psnr)
        eval_ssim_list.append(ssim)
        eval_lpips_list.append(lpips_loss)

    mean_mse = np.mean(eval_mse_list)
    mean_psnr = np.mean(eval_psnr_list)
    mean_ssim = np.mean(eval_ssim_list)
    mean_lpips = np.mean(eval_lpips_list)
    print('--------------------------')
    print('Mean MSE: {0:.2f}, PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(mean_mse, mean_psnr,
                                                                                  mean_ssim, mean_lpips))
    return


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, optimizer_st, model, focal_net, pose_param_net, st_net,
                    my_devices, args, rgb_act_fn):
    model.train()
    focal_net.train()
    pose_param_net.train()
    st_net.train()

    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    ids = np.arange(N_img)
    np.random.shuffle(ids)

    for i in ids:

        fxfy = focal_net(0)
        st = st_net(0)
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        ray_dir_cam = ray_dir_cam.reshape(H,W,-1).to(my_devices)
        img = scene_train.imgs[i].to(my_devices) 
        c2w = pose_param_net(i) 

        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  
        ray_selected_cam = ray_dir_cam[r_id][:, c_id] 
        img_selected = img[r_id][:, c_id] 
        render_result = model_render_image(c2w, ray_selected_cam, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy, st, 
                                           model, True, 0.0, args, rgb_act_fn) 
        rgb_rendered = render_result['rgb']  

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  
        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_st.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()
        optimizer_st.zero_grad()
        L2_loss_epoch.append(L2_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    exp_root_dir = Path(os.path.join('./logs', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = Path(os.path.join(experiment_dir, 'img_out'))
    img_out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy('./models/model.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/UNeLF.py', experiment_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    writer = SummaryWriter(log_dir=str(experiment_dir))

    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      data_type='train',
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted)

    scene_eval = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      data_type='val',
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted)

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))
    print(scene_train.near)
    print(scene_train.far)
    eval_c2ws = torch.eye(4).unsqueeze(0).float() 

    model = UNeLF()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    st_net = scale(scene_train.H, scene_train.W, True, args.st_scale)
    if args.multi_gpu:
        st_net = torch.nn.DataParallel(st_net).to(device=my_devices)
    else:
        st_net = st_net.to(device=my_devices)

    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    pose_param_net = pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, my_devices, args.rows, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)

    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)
    optimizer_st = torch.optim.Adam(st_net.parameters(), lr=args.st_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
                                                          gamma=args.nerf_lr_gamma)
    scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
                                                           gamma=args.focal_lr_gamma)
    scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
                                                          gamma=args.pose_lr_gamma)
    scheduler_st = torch.optim.lr_scheduler.MultiStepLR(optimizer_st, milestones=args.st_milestones,
                                                           gamma=args.st_lr_gamma)

    best_psnr = torch.tensor(0.0).to(device=my_devices)

    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, optimizer_st, 
                                             model, focal_net, pose_param_net, st_net, my_devices, args, rgb_act_fn)
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()
        scheduler_st.step()

        train_psnr = mse2psnr(train_L2_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        if epoch_i % args.eval_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net, st_net, my_devices, args, epoch_i, writer, rgb_act_fn)
                fxfy = focal_net(0)

                if train_psnr > best_psnr:
                    best_psnr = train_psnr
                    save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='best_nerf')
                    save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='best_focal')
                    save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='best_pose')
                    save_checkpoint(epoch_i, st_net, optimizer_st, experiment_dir, ckpt_name='best_st')

    with torch.no_grad():
        model = load_ckpt_to_net(os.path.join(experiment_dir, 'best_nerf.pth'), model, map_location=my_devices)
        focal_net = load_ckpt_to_net(os.path.join(experiment_dir, 'best_focal.pth'), focal_net, map_location=my_devices)
        pose_param_net = load_ckpt_to_net(os.path.join(experiment_dir, 'best_pose.pth'), pose_param_net, map_location=my_devices)
        st_net = load_ckpt_to_net(os.path.join(experiment_dir, 'best_st.pth'), st_net, map_location=my_devices)

        train_c2ws = []
        for i in range(0,scene_train.N_imgs):
            train_c2ws.append(pose_param_net(i))
        poses = torch.cat(train_c2ws, dim=0).reshape(-1,4,4).to(device=my_devices)

        eval_c2ws = []
        generatePosesV = []
        generatePosesH = []
        generatePosesH4 = []
        for i in range(0, poses.shape[0]):
            if i%5==4:
                generatePosesH.append(poses[i,:,:])
                continue
            lr_poses = poses_avg(poses[i:i + 2,:,:])
            generatePosesH.append(poses[i,:,:])
            generatePosesH.append(lr_poses)
            generatePosesH4.append(lr_poses)
        generatePosesH = torch.cat(generatePosesH, dim=0).reshape(-1,4,4)
        generatePosesH4 = torch.cat(generatePosesH4, dim=0).reshape(-1,4,4)

        for i in range(0, 36):
            tmp = []
            tmp.append(generatePosesH[i,:,:])
            tmp.append(generatePosesH[i+9,:,:])
            tmp = torch.cat(tmp, dim=0).reshape(-1,4,4)
            ud_poses = poses_avg(tmp)
            generatePosesV.append(ud_poses)
        generatePosesV = torch.cat(generatePosesV, dim=0).reshape(-1,4,4)

        for i in range (0, 4):
            for j in range(0,4):
                eval_c2ws.append(generatePosesH4[4*i+j,:,:])
            for j in range(0,9):
                eval_c2ws.append(generatePosesV[9*i+j,:,:])
        for j in range(0,4):
            eval_c2ws.append(generatePosesH4[4*4+j,:,:])
        eval_c2ws = torch.cat(eval_c2ws, dim=0).reshape(-1,4,4)
        eval_all(scene_eval, eval_c2ws, model, focal_net, pose_param_net, st_net, my_devices, args, rgb_act_fn, img_out_dir)
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)

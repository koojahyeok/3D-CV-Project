import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio
import json
import pickle

from utils.comp_ray_dir import comp_ray_dir_cam
from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4


def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(image_dir, img_ids, new_h, new_w):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    img_list = resize_imgs(img_list, new_h, new_w)
    return img_list, img_names


def load_split(scene_dir, img_dir, data_type, num_img_to_load, skip, c2ws,
               H, W, load_img):
    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join(scene_dir, data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    N_imgs = img_ids.shape[0]

    # use img_ids to select camera poses
    c2ws = c2ws[img_ids]  # (N, 3, 4)

    # load images
    if load_img:
        imgs, img_names = load_imgs(img_dir, img_ids, H, W)  # (N, H, W, 3) torch.float32
    else:
        imgs, img_names = None, None

    result = {
        'c2ws': c2ws,  # (N, 3, 4) np.float32
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'img_ids': img_ids,  # (N, ) np.int
    }
    return result


def read_meta(in_dir, param_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    
    img_cnt = len(os.listdir(os.path.join(in_dir, 'images')))
    print(f'Number of image : {img_cnt}')

    # get extrinsic parameter
    c2ws = np.zeros((img_cnt, 3, 4))
    for i in range(img_cnt):
        c2ws[i] = np.load(os.path.join(os.path.join(in_dir, param_dir), '{0:03d}_rt.npy'.format(i)))

    # get intrinsic parameter
    K = np.zeros((img_cnt, 3, 3))
    for i in range(img_cnt):
        K[i] = np.load(os.path.join(os.path.join(in_dir, param_dir), '{0:03d}_k.npy'.format(i)))

    # we need whole size      
    H = K[0, 1, 2] * 2              # scalar
    W = K[0, 0, 2] * 2              # scalar

    # focal = np.mean(K[:, 0, 0])              # (N, 1)
    focal = torch.FloatTensor(K[:, 0, 0])  # (N, 1)

    # get depth (near, far)
    bounds = np.zeros((img_cnt, 2)) #(N, 2)
    for i in range(img_cnt):
        with open(os.path.join(os.path.join(in_dir, param_dir), '{0:03d}.pkl'.format(i)), 'rb') as f:
            loaded = pickle.load(f)
        near = loaded['near']
        far = loaded['far']
        bounds[i] = np.array([near, far])

    # I dont know we have to do this code.... experimentely go
    c2ws, pose_avg = center_poses(c2ws)             # pose_avg @ c2ws -> centered c2ws

    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor
    
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal,     # (N, 1) np
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


class DataLoaderWithBLENDER:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, camera_param_name, load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.camera_param_name = camera_param_name
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.camera_param_name, self.use_ndc)
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = meta['focal']
        self.total_N_imgs = self.c2ws.shape[0]

        if self.res_ratio > 1:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio

        self.near = 0.0
        self.far = 1.0

        '''Load train/val split'''
        split_results = load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load,
                                   self.skip, self.c2ws, self.H, self.W, self.load_img)
        self.c2ws = split_results['c2ws']  # (N, 4, 4) np.float32
        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']
        self.img_ids = split_results['img_ids']  # (N, ) np.int
        self.focal = self.focal[self.img_ids]

        # generate cam ray dir.
        self.ray_dir_cam = comp_ray_dir_cam(self.H, self.W, self.focal)  # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float()  # (N, 4, 4) torch.float32
        # self.ray_dir_cam = self.ray_dir_cam.float()  # (H, W, 3) torch.float32
        self.ray_dir_cam = torch.concat([torch.FloatTensor(i).unsqueeze(0) for i in self.ray_dir_cam], dim=0)
        print(self.ray_dir_cam.shape)


if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataLoaderWithBLENDER(base_dir='/your/data/path',
                                 scene_name=scene_name,
                                 data_type='train',
                                 res_ratio=8,
                                 num_img_to_load=-1,
                                 skip=1,
                                 use_ndc=use_ndc)
    

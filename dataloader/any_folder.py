import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn.functional as F

def resize_imgs(imgs, new_h, new_w):
    imgs = imgs.permute(0, 3, 1, 2) 
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear') 
    imgs = imgs.permute(0, 2, 3, 1) 

    return imgs

def load_split(scene_dir, img_dir, data_type, num_img_to_load, skip, load_img):
    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join('data/', data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
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

    if load_img:
        imgs, img_names = load_imgs(img_dir, img_ids) 
    else:
        imgs, img_names = None, None
    H, W = imgs[0].shape[0], imgs[0].shape[1]
    result = {
        'imgs': imgs,  
        'img_names': img_names, 
        'N_imgs': N_imgs,
        'img_ids': img_ids,  
        'H': H,
        'W': W,
    }
    return result


def load_imgs(image_dir, img_ids):
    img_names = np.array(sorted(os.listdir(image_dir))) 
    img_names = img_names[img_ids]  
    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]
        img_list.append(img)
    img_list = np.stack(img_list)  
    img_list = torch.from_numpy(img_list).float() / 255  

    return img_list, img_names

class DataLoaderAnyFolder:
    def __init__(self, base_dir, scene_name, res_ratio, num_img_to_load, data_type, start, end, skip, load_sorted, load_img=True):

        self.base_dir = base_dir
        self.scene_name = scene_name
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.start = start
        self.end = end
        self.skip = skip
        self.load_sorted = load_sorted
        self.load_img = load_img
        self.data_type = data_type

        self.imgs_dir = os.path.join(self.base_dir, self.scene_name, 'images')
        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        image_data = load_split(self.scene_dir, self.imgs_dir, self.data_type, self.num_img_to_load, self.skip, self.load_img)


        self.imgs = image_data['imgs']
        self.img_names = image_data['img_names']
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        self.img_ids = image_data['img_ids']

        # always use ndc
        self.near = 0.0
        self.far = 1.0

        if self.res_ratio > 1:
            self.H = self.ori_H // self.res_ratio
            self.W = self.ori_W // self.res_ratio
        else:
            self.H = self.ori_H
            self.W = self.ori_W

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W)

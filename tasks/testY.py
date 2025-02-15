

import os
import imageio
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
from tqdm import tqdm

def calculate_psnr_y_channel(img1, img2):

    img1_y = img1[:, :, 0]
    img2_y = img2[:, :, 0]

    mse = np.mean((img1_y - img2_y) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim_y_channel(img1, img2):

    img1_y = img1[:, :, 0]
    img2_y = img2[:, :, 0]

    return ssim(img1_y, img2_y, data_range=img1_y.max()-img1_y.min())


def calculate_avg_psnr_ssim_y_channel(images1, images2):
    psnr_values = []
    ssim_values = []

    for img1, img2 in zip(images1, images2):
        psnr_val = calculate_psnr_y_channel(img1, img2)
        ssim_val = calculate_ssim_y_channel(img1, img2)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y

logfolder = "/home/zmybuaa/UNeLF/logs"
datafolder = "/home/zmybuaa/UNeLF/data"
scenes = sorted(os.listdir(logfolder))
print(scenes)

scenes = ['Amethyst', 'Beans', 'Bracelet', 'Bulldozer', 'Bunny', 'Chess', 'Flowers', 'Knights', 'Portrait', 'Tarot_small', 'Treasure', 'Truck', 'bedroom', 'bicycle', 'boxes', 'cotton', 'dino', 'herbs', 'origami', 'sideboard']

all_psnr = []
all_ssim = []

for scene in scenes:
    scenePath = os.path.join(datafolder, scene)
    img_dir = os.path.join(scenePath, 'images')
    img_ids = np.loadtxt(os.path.join(scenePath, 'val_ids.txt'), dtype=np.int32, ndmin=1)
    img_names = np.array(sorted(os.listdir(img_dir))) 
    img_names = img_names[img_ids] 
    img_paths = [os.path.join(img_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = rgb2ycbcr(imageio.imread(p)[:, :, :3]) 
        img_list.append(img)
    
    resultfolders = os.listdir(os.path.join(logfolder, scene))
    
    for resultfolder in resultfolders:
        resultPath = os.path.join(logfolder, scene, resultfolder, 'img_out')
        result_names = np.array(sorted(os.listdir(resultPath)))
        res_paths = [os.path.join(resultPath, n) for n in result_names]
        res_list = []
        for p in tqdm(res_paths):
            res = rgb2ycbcr(imageio.imread(p)[:, :, :3]) 
            res_list.append(res)
        avg_psnr, avg_ssim = calculate_avg_psnr_ssim_y_channel(img_list, res_list)
        print(f'{scene} PSNR: {avg_psnr}')
        print(f'{scene} SSIM: {avg_ssim}')
        all_psnr.append(avg_psnr)
        all_ssim.append(avg_ssim)
all_psnr = np.mean(all_psnr)
all_ssim = np.mean(all_ssim)
print(f'All PSNR: {all_psnr}')
print(f'All SSIM: {all_ssim}')
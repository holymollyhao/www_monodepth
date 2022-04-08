import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import skimage.transform
from PIL import Image
import imageio

from main_monodepth_pytorch import Model

DATA_PATH = '/home/twkim/git/monodepth/datasets/VIDEO/KITTI/tracking/training/image_02'
OUTPUT_PATH = '/home/twkim/git/monodepth/output/tracking/training/image_02'
OUTPUT_PATH_IMG = '/home/twkim/git/rain_rendering/data/source/kitti/data_object/training'
print(os.listdir('/home/twkim/git/monodepth/datasets/VIDEO/KITTI/tracking/training/'))

for dir_name in os.listdir(DATA_PATH):
    print(f'starting directory : {dir_name}')
    if not os.path.isdir(os.path.join(OUTPUT_PATH, dir_name)):
            print(f"making directory : {os.path.join(OUTPUT_PATH, dir_name)}")
            os.makedirs(os.path.join(OUTPUT_PATH, dir_name))
    dict_parameters_test = edict({'data_dir': os.path.join(DATA_PATH, dir_name),
                                  'model_path':'/home/twkim/git/monodepth/models/monodepth_resnet18_001.pth',
                                  'output_directory':os.path.join(OUTPUT_PATH, dir_name),
                                  'input_height':256,
                                  'input_width':512,
                                  'model':'resnet18_md',
                                  'pretrained':False,
                                  'mode':'test',
                                  'device':'cuda:0',
                                  'input_channels':3,
                                  'num_workers':4,
                                  'use_multiple_gpu':False})
    model_test = Model(dict_parameters_test)
    model_test.test()

    disp = np.load(os.path.join(dict_parameters_test.output_directory, 'disparities_pp.npy'))  # Or disparities.npy for output without post-processing

    for i in range(disp.shape[0]):
        left_image = Image.open(os.path.join(dict_parameters_test.data_dir, '000000.png'))
        print(left_image.size)
        if not os.path.isdir(os.path.join(OUTPUT_PATH_IMG,dir_name,'image_2', 'depth')):
            print(f"making directory : {os.path.join(OUTPUT_PATH, dir_name)}")
            os.makedirs(os.path.join(OUTPUT_PATH_IMG,dir_name,'image_2', 'depth'))
        print(disp.shape)
        disp_to_img = skimage.transform.resize(disp[i].squeeze(), [left_image.size[1], left_image.size[0]], mode='constant')
        print(disp_to_img.shape)
        print("\n")
        print(os.path.join(OUTPUT_PATH_IMG,dir_name,'depth', str(i).zfill(6)+'.png'))
        imageio.imwrite(os.path.join(OUTPUT_PATH_IMG,dir_name,'image_2','depth', str(i).zfill(6)+'.png'), disp_to_img.astype(np.uint16))

    print(f'finishing directory : {dir_name}')

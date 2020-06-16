"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., bicycle_gan, pix2pix, test) and
different datasets (with option '--dataset_mode': e.g., aligned, single).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a BiCycleGAN model:
        python train.py --dataroot ./datasets/facades --name facades_bicyclegan --model bicycle_gan --direction BtoA
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
"""
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from skimage import io, transform, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import time
import cv2

from os import listdir
from os.path import isfile, join
import re

sys.path.append('./geo_net')
sys.path.append('./bicycle_net')

from model import S2SModel
from dataset import MyDataLoader
from bicycle_net.options.test_options import TeseOptions

def test():
    # set options
    opt = TeseOptions().parse()   # get training options    
    opt.dataroot = './data/test'
    opt.resultroot = './data/result'
    opt.checkpoints_dir = './checkpoints'
    opt.gpu_ids = [0]
    opt.batch_size = 4
    opt.coarse = False    
    opt.nz = 32
    opt.load_size_w = 512
    opt.load_size_h = 256
    opt.crop_size_w = 256
    opt.crop_size_h = 256
    opt.ngf_S3 = 96
    opt.nef_S3 = 96
    opt.ndf_S3 = 96
    opt.ngf_S12 = 64
    opt.nef_S12 = 64
    opt.ndf_S12 = 64    
    opt.input_nc = 3
    opt.is_train = False

    if not os.path.exists(opt.resultroot):
        os.makedirs(opt.resultroot)        
        
    # load data      
    dataset_test = MyDataLoader(root_dir=opt.dataroot, train=opt.is_train, coarse=opt.coarse)
    data_loader = DataLoader(dataset_test,batch_size=opt.batch_size,
                                shuffle=False, num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    # load model
    model = S2SModel(opt)    
    model.load_networks(-1)

    # do testing
    for idx_batch, data_batch in enumerate(data_loader):
        print(idx_batch)
        model.set_input(data_batch)
        model.forward()
        sate_L = model.sate_fake_L.detach().cpu()
        sate_D = model.sate_fake_D.detach().cpu()
        proj_L = model.net_stage2.proj_L.detach().cpu()
        proj_D = model.net_stage2.proj_D.detach().cpu()
        str_S = model.street_fake_S.detach().cpu()
        str_R = model.street_fake_R.detach().cpu()        
        n,c,h,w = proj_L.size()
        for i in range(0, n):
            sate_label = sate_L[i,:,:,:] * 0.5 + 0.5
            sate_depth = sate_D[i,:,:,:] * 0.5 + 0.5
            label = proj_L[i,:,:,:] * 0.5 + 0.5
            depth = proj_D[i,:,:,:] * 0.5 + 0.5
            str_sem = str_S[i,:,:,:] * 0.5 + 0.5            
            str_rgb = str_R[i,:,:,:] * 0.5 + 0.5  
            img_id = data_batch['img_id'][i]

            # save image
            path_sate_label = opt.resultroot + '/' + img_id + '_pred_sate_label.png'
            path_sate_depth = opt.resultroot + '/' + img_id + '_pred_sate_depth.png'
            path_depth = opt.resultroot + '/' + img_id + '_pred_proj_dis.png'
            path_label = opt.resultroot + '/' + img_id + '_pred_proj_label.png'            
            path_sem = opt.resultroot + '/' + img_id + '_pred_str_sem.png'
            path_rgb = opt.resultroot + '/' + img_id + '_pred_str_rgb.png'

            torchvision.utils.save_image(sate_label.float(), path_sate_label)
            torchvision.utils.save_image(sate_depth.float(), path_sate_depth)
            torchvision.utils.save_image(label.float(), path_label)
            torchvision.utils.save_image(depth.float(), path_depth)
            torchvision.utils.save_image(str_sem.float(), path_sem)
            torchvision.utils.save_image(str_rgb.float(), path_rgb)            

if __name__ == "__main__":    
    test()
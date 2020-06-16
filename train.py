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
from bicycle_net.options.train_options import TrainOptions


def train():
    # set options
    opt = TrainOptions().parse()   # get training options    
    opt.dataroot = './data/train'    
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
    opt.is_train = True

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)        
        
    # load data      
    dataset_train = MyDataLoader(root_dir=opt.dataroot, train=opt.is_train, coarse=opt.coarse)
    data_loader = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=True, num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    # load model
    model = S2SModel(opt)    
    #model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.dataroot+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader):
            model.set_input(data_batch)
            model.optimize_parameters()       
            str_disp = 'epoch: ' + str(epoch) + ', batch: ' + str(idx_batch) + ', stage1 G loss: ' + str(model.net_stage1.loss_G.data) \
            + ', stage2 G loss: ' + str(model.net_stage2.loss_G.data) \
            + ', stage3 G loss: ' + str(model.net_stage3.loss_G.data) 
            print(str_disp)            
        file.write(str_disp + '\n')
        file.close()

        # save
        if epoch%5 ==0:
            model.save_networks(epoch)

if __name__ == "__main__":    
    train()
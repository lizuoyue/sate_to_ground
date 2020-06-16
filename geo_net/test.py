import numpy as np
import os
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

from model import R2DModel, D2LModel, L2RModel, DLRModel, RDLRModel, DLLModel
from utils import Option
from dataset import R2DDataLoader, D2LDataLoader, L2RDataLoader, DLRDataLoader, RDLRDataLoader, DLLDataLoader
from geo_process_layer import depth2voxel, voxel2pano

root = 'F:/DevelopCenter/CNN/ICCV2019'

def test_R2D():
# set options
    opt = Option()
    opt.root_dir = root+'/dataset/R2D'
    opt.checkpoints_dir = root+'/checkpoints/R2D'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True

    # load data  
    root_dir_test = opt.root_dir + '/test'
    dataset_test = R2DDataLoader(root_dir=root_dir_test, train=True, coarse=opt.coarse)
    data_loader_test = DataLoader(dataset_test, batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = R2DModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch)
        model.forward()
        fake_L = model.fake_L.detach().cpu()
        fake_D = model.fake_D.detach().cpu()
        n,c,h,w = fake_L.size()
        for i in range(0, n):
            label = fake_L[i,:,:,:] * 0.5 + 0.5
            depth = fake_D[i,:,:,:] * 0.5 + 0.5
            img_id = data_batch['img_id'][i]
            # save image
            path_depth = 'F:/' + img_id + '_pred_depth.png'
            path_label = 'F:/' + img_id + '_label.png'
            #torchvision.utils.save_image(depth.float(), path_depth)
            torchvision.utils.save_image(label.float(), path_label)
            torchvision.utils.save_image(depth.float(), path_depth)

def test_D2L():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/test'
    opt.checkpoints_dir = root+'/checkpoints/D2L'
    opt.result_dir = opt.root_dir
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.is_train = False
    opt.fine_tune_sidewalk = False

    # load data  
    root_dir_train = opt.root_dir
    dataset_train = D2LDataLoader(root_dir=root_dir_train, train=opt.is_train, coarse=opt.coarse,fine_tune_sidewalk=opt.fine_tune_sidewalk)
    data_loader_test = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = D2LModel()
    model.initialize(opt)
    model.load_networks(50)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch, 0)
        model.forward()
        fake_S = model.fake_S.detach().cpu()
        n,c,h,w = fake_S.size()
        for i in range(0, n):
            sem = fake_S[i,:,:,:] * 0.5 + 0.5
            img_id = data_batch['img_id'][i]
            # save image
            path_sem = root_dir_train + '/' + img_id + '_pred_sem_wo_mask.png'
            #torchvision.utils.save_image(depth.float(), path_depth)
            torchvision.utils.save_image(sem.float(), path_sem)

def test_L2R():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/'
    opt.checkpoints_dir = root+'/checkpoints/L2R'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.is_train = False

    # load data  
    root_dir_train = opt.root_dir + '/test3000'
    dataset_train = L2RDataLoader(root_dir=root_dir_train, train=False, coarse=opt.coarse)
    data_loader_test = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = L2RModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch)
        model.forward()
        fake_R = model.fake_R.detach().cpu()
        n,c,h,w = fake_R.size()
        for i in range(0, n):
            rgb = fake_R[i,:,:,:] * 0.5 + 0.5
            img_id = data_batch['img_id'][i]
            # save image
            path_rgb = root_dir_train + '/' + img_id + '_pred_rgb_dll.png'
            #torchvision.utils.save_image(depth.float(), path_depth)
            torchvision.utils.save_image(rgb.float(), path_rgb)


def test_DLR():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/'
    opt.checkpoints_dir = root+'/checkpoints/DLR'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.is_train = False

    # load data  
    root_dir_train = opt.root_dir + '/test'
    dataset_train = DLRDataLoader(root_dir=root_dir_train, train=False, coarse=opt.coarse)
    data_loader_test = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = DLRModel()
    model.initialize(opt)
    model.load_networks(15)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch,0)
        model.forward()
        fake_R = model.fake_R.detach().cpu()
        fake_L = model.fake_L.detach().cpu()
        n,c,h,w = fake_R.size()
        for i in range(0, n):
            rgb = fake_R[i,:,:,:] * 0.5 + 0.5
            label = fake_L[i,:,:,:] * 0.5 + 0.5

            img_id = data_batch['img_id'][i]
            # save image
            path_rgb = opt.root_dir + '/test/' + img_id + '_pred_rgb_finetune2.png'
            path_label = opt.root_dir + '/test/' + img_id + '_pred_sem_finetune2.png'

            #torchvision.utils.save_image(depth.float(), path_depth)
            torchvision.utils.save_image(rgb.float(), path_rgb)
            torchvision.utils.save_image(label.float(), path_label)


def test_RDLR():
    t = '5'
    # set options
    opt = Option()
    opt.root_dir = 'D:/permanent/aligned_2k/test_augment/test_'+t
    opt.checkpoints_dir = 'C:/Users/lu.2037/Downloads/ICCV2019/checkpoints/RDLR'
    root_result = 'D:/permanent/aligned_2k/test_augment/test_'+t
    opt.gpu_ids = [0]
    opt.batch_size = 4
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.is_train = False

    # load data  
    root_dir_train = opt.root_dir
    dataset_train = RDLRDataLoader(root_dir=root_dir_train, train=False, coarse=opt.coarse)
    data_loader_test = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = RDLRModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch,0)
        model.forward()
        fake_R = model.fake_street_R.detach().cpu()
        fake_L = model.fake_street_L.detach().cpu()
        fake_sate_D = model.fake_sate_D.detach().cpu()
        fake_sate_L = model.fake_sate_L.detach().cpu()
        fake_proj_dis = model.proj_D.detach().cpu()

        n,c,h,w = fake_R.size()
        for i in range(0, n):
            rgb = fake_R[i,:,:,:] * 0.5 + 0.5
            label = fake_L[i,:,:,:] * 0.5 + 0.5
            sate_depth = fake_sate_D[i,:,:,:] * 0.5 + 0.5
            sate_label = fake_sate_L[i,:,:,:] * 0.5 + 0.5
            proj_depth = fake_proj_dis[i,:,:,:] * 0.5 + 0.5
            img_id = data_batch['img_id'][i]
            # save image
            tt = "_0"+t
            path_depth = root_result + '/' + img_id + '_pred_depth'+tt+'.png'
            path_sate_label = root_result + '/' + img_id + '_pred_label'+tt+'.png'
            path_rgb = root_result + '/' + img_id + '_pred_rgb'+tt+'.png'
            path_label = root_result + '/' + img_id + '_pred_sem'+tt+'.png'
            path_proj_dis = root_result + '/' + img_id + '_proj_dis'+tt+'.png'

            torchvision.utils.save_image(sate_depth.float(), path_depth)
            # torchvision.utils.save_image(sate_label.float(), path_sate_label)
            torchvision.utils.save_image(rgb.float(), path_rgb)
            torchvision.utils.save_image(label.float(), path_label)
            torchvision.utils.save_image(proj_depth.float(), path_proj_dis)


def test_DLL():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset'
    opt.checkpoints_dir = root+'/checkpoints/DLL'
    opt.gpu_ids = [0]
    opt.batch_size = 8
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.is_train = False

    # load data  
    root_dir_train = opt.root_dir + '/test3000'
    dataset_train = DLLDataLoader(root_dir=root_dir_train, train=False, coarse=opt.coarse)
    data_loader_test = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = DLLModel()
    model.initialize(opt)
    model.load_networks(30)

    # do testung
    for idx_batch, data_batch in enumerate(data_loader_test):
        print(idx_batch)
        model.set_input(data_batch)
        model.forward()
        fake_S = model.fake_S.detach().cpu()
        n,c,h,w = fake_S.size()
        for i in range(0, n):
            label = fake_S[i,:,:,:] * 0.5 + 0.5
            label = label.numpy()
            label_rgb = np.zeros([256,256,3]).astype(np.uint8)
            label_rgb[:,:,2] = label[0,:,:]*255
            label_rgb[:,:,1] = label[1,:,:]*255
            label_rgb[:,:,0] = label[2,:,:]*255
            label_rgb = cv2.resize(label_rgb,(512,256))

            img_id = data_batch['img_id'][i]
            # save image
            path_label = root_dir_train + '/' + img_id + '_pred_sem_dll.png'
            #torchvision.utils.save_image(label_rgb, path_label)
            cv2.imwrite(path_label,label_rgb)            


if __name__ == '__main__':
    #test_R2D()  # satellite rgb to depth + semantic
    #test_D2L()  # street depth to street semantic
    #test_L2R()  # street semantic to street rgb
    #test_DLR()  # street depth to semantic to rgb, 3 in 1
    test_RDLR()  # satellite rgb to street depth to semantic to rgb, 4 in 1
    #test_DLL()  # satellite depth to satellite semantic to street semantic, 3 in 1

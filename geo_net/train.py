import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

from model import R2DModel, D2LModel, L2RModel, DLRModel, RDLRModel, DLLModel
from utils import Option
from dataset import R2DDataLoader, D2LDataLoader, L2RDataLoader, DLRDataLoader, RDLRDataLoader, DLLDataLoader

root = 'F:/DevelopCenter/CNN/ICCV2019'

def train_R2D():
    # set options
    opt = Option()
    opt.root_dir = 'D:/permanent/aligned_2k/train_R2D'
    opt.checkpoints_dir = 'C:/Users/lu.2037/Downloads/ICCV2019/checkpoints/R2D'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True

    # load data  
    root_dir_train = opt.root_dir
    dataset_train = R2DDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = R2DModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) + ', train loss_G_Loss1: ' + str(model.loss_G_Loss1.data) + 
            ', train loss_G_Loss2: ' + str(model.loss_G_Loss2.data) )
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss1: ' + str(model.loss_G_Loss1.data) + 
            ', train loss_G_Loss2: ' + str(model.loss_G_Loss2.data) + '\n')
        file.close()

        # save
        if epoch%5 ==0:
            model.save_networks(epoch)

def train_D2L():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/D2L'
    opt.checkpoints_dir = root+'/checkpoints/D2L'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.fine_tune_sidewalk = False

    # load data  
    root_dir_train = opt.root_dir + '/train'
    dataset_train = D2LDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse, fine_tune_sidewalk=opt.fine_tune_sidewalk)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    # load model
    model = D2LModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch, epoch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data) )
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data) + '\n')
        file.close()

        # save
        if epoch % 5 == 0:
            model.save_networks(epoch)


def train_L2R():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/L2R'
    opt.checkpoints_dir = root+'/checkpoints/L2R'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True

    # load data  
    root_dir_train = opt.root_dir + '/train'    
    dataset_train = L2RDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    print(opt)

    # load model
    model = L2RModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data) 
            + ', train loss_D_Loss: ' + str(model.loss_D.data) )
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data)
        + ', train loss_D_Loss: ' + str(model.loss_D.data)  + '\n')
        file.close()

        # save
        if epoch % 10 == 0:
            model.save_networks(epoch)


def train_DLR():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/DLR/'
    opt.checkpoints_dir = root+'/checkpoints/DLR'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.learning_rate = 1e-3	# learning rage

    # load data  
    root_dir_train = opt.root_dir + '/train'
    dataset_train = DLRDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    print(opt)

    # load model
    model = DLRModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch, epoch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) 
            #+ ', train loss_G1_Loss: ' + str(model.loss_G_L1.data) 
            + ', train loss_G2_Loss: ' + str(model.loss_G_L2.data)
            + ', train loss_GAN_Loss: ' + str(model.loss_G_GAN.data) 
            + ', train loss_D_Loss: ' + str(model.loss_D.data) )
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data)
        + ', train loss_D_Loss: ' + str(model.loss_D.data)  + '\n')
        file.close()

        # save
        if epoch % 5 == 0:
            model.save_networks(epoch)


def train_RDLR():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/RDLR/'
    opt.checkpoints_dir = root+'/checkpoints/RDLR'
    opt.gpu_ids = [0]
    opt.batch_size = 8
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    opt.learning_rate = 2e-4	# learning rage

    # load data  
    root_dir_train = opt.root_dir + '/train'
    dataset_train = RDLRDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    print(opt)

    # load model
    model = RDLRModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch, epoch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data) 
            + ', train loss_D_Loss: ' + str(model.loss_D.data) )
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data)
        + ', train loss_D_Loss: ' + str(model.loss_D.data)  + '\n')
        file.close()

        # save
        if epoch % 10 == 0:
            model.save_networks(epoch)


def train_DLL():
    # set options
    opt = Option()
    opt.root_dir = root+'/dataset/RDLR/'
    opt.checkpoints_dir = root+'/checkpoints/DLL'
    opt.gpu_ids = [0]
    opt.batch_size = 16
    opt.coarse = False
    opt.pool_size = 0
    opt.no_lsgan = True
    #opt.learning_rate = 2e-3	# learning rage

    # load data  
    root_dir_train = opt.root_dir + '/train'
    dataset_train = DLLDataLoader(root_dir=root_dir_train, train=True, coarse=opt.coarse)
    data_loader_train = DataLoader(dataset_train,batch_size=opt.batch_size,
                                shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    print(opt)

    # load model
    model = DLLModel()
    model.initialize(opt)
    model.load_networks(-1)

    # do training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        file = open(opt.root_dir+'/logs.txt', 'a')
        for idx_batch, data_batch in enumerate(data_loader_train):
            print(idx_batch)
            model.set_input(data_batch)
            model.optimize_parameters()
            print('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data))
        file.write('epoch: ' + str(epoch) + ', train loss_G_Loss: ' + str(model.loss_G.data)+ '\n')
        file.close()

        # save
        if epoch % 10 == 0:
            model.save_networks(epoch)



if __name__ == '__main__':
    train_R2D()  # satellite rgb to depth + semantic
    #train_D2L()  # street depth to street semantic
    #train_L2R()  # street semantic to street rgb
    #train_DLR()  # street depth to semantic to rgb, 3 in 1
    #train_RDLR()  # satellite rgb to street depth to semantic to rgb, 4 in 1
    #train_DLL()  # satellite depth to satellite semantic to street semantic, 3 in 1

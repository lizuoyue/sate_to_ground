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

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, utils
import torch.nn.functional as F

from skimage import io, transform, color

import cv2
import numpy
import time

# geo net
import geo_net.model as geo_model
import geo_net.networks as geo_networks

# bicycle net
import bicycle_net.models
import bicycle_net.models.bicycle_gan_model as bicycle_model


class S2SModel:
    def name(self):
        return 'S2SModel'

    def __init__(self, opt):        
        super().__init__()
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode        

        # stage1 model
        self.net_stage1 = geo_model.R2DL(opt)

        # stage2 model
        self.net_stage2 = geo_model.DL2S(opt)

        # stage3 model
        self.net_stage3 = bicycle_model.BiCycleGANModel(opt)

        if self.is_train:
            # define loss functions
            self.criterion_Geo_GAN = geo_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterion_Geo_L1 = torch.nn.L1Loss()
            self.criterion_Bicycle_Z = torch.nn.L1Loss()
            self.criterion_Bicycle_L1 = torch.nn.L1Loss()
            self.criterion_Bicycle_GAN = bicycle_net.models.networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)

            # initialize optimizers
            self.optimizers = []

            self.optimizer_S1_G = torch.optim.Adam(self.net_stage1.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S1_G)

            self.optimizer_S2_G = torch.optim.Adam(self.net_stage2.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S2_G)

            # initialize optimizers            
            self.optimizer_S3_G = torch.optim.Adam(self.net_stage3.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S3_G)
            if use_E:
                self.optimizer_S3_E = torch.optim.Adam(self.net_stage3.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_S3_E)
            if use_D:
                self.optimizer_S3_D = torch.optim.Adam(self.net_stage3.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_S3_D)
            if use_D2:
                self.optimizer_S3_D2 = torch.optim.Adam(self.net_stage3.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_S3_D2)


    def set_input(self, input):
        self.sate_real_R = input['sate_R'].to(self.device)
        if self.is_train:      
            self.sate_real_D = input['sate_D'].to(self.device)
            self.sate_real_L = input['sate_L'].to(self.device)                  
            self.street_real_S = input['street_S'].to(self.device)
            self.street_real_M = input['street_M'].to(self.device)
            self.street_real_R = input['street_R'].to(self.device)
        self.img_id = input['img_id']
        self.ori = input['ori']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cvt_geo2bicycle(self):
        colors_3d = [[ 70, 130, 180], [ 70,  70,  70], [128,  64, 127], [106, 142,  34], [243,  36, 232]]
        num_colors = len(colors_3d)

        # step 1    
        if self.is_train:
            n,d,h,w = self.street_real_S.size()
        else:
            n,d,h,w = self.street_fake_S.size()

        dis_gt = torch.zeros([n,num_colors,h,w])
        for i in range(len(colors_3d)):
            c = colors_3d[i]
            if self.is_train:                
                sem_gt = torch.from_numpy(numpy.array(c)).view(1, 3, 1, 1).expand(n, d, h, w).cuda()
                sem_gt = (sem_gt/255.0-0.5)/0.5                
                dis_gt[:,i,:,:] = torch.sum(self.street_real_S - sem_gt, dim=1)
            else:                
                sem_gt = torch.from_numpy(numpy.array(c)).view(1, 3, 1, 1).expand(n, d, h, w).cuda()
                sem_gt = (sem_gt/255.0-0.5)/0.5                
                dis_gt[:,i,:,:] = torch.sum((self.street_fake_S - sem_gt)**2, dim=1)                
        single_sem = torch.argmin(dis_gt, dim=1).view(n*1*h*w)
        
        # step 2
        colors_2d = [[255, 255],[  0, 127],[127,   0],[  0, 255],[255,   0]]                
        merge_1 = torch.zeros([n*1*h*w])
        merge_2 = torch.zeros([n*1*h*w])        
        merge_3 = self.proj_fake_depth.view(n*1*h*w)        
        for i in range(num_colors):
            merge_1[single_sem==i] = colors_2d[i][0]
            merge_2[single_sem==i] = colors_2d[i][1]
        merge_1 = merge_1.view(n, 1, h, w).cuda()
        merge_1 = (merge_1/255.0-0.5)/0.5  
        merge_2 = merge_2.view(n, 1, h, w).cuda()
        merge_2 = (merge_2/255.0-0.5)/0.5  
        merge_3 = merge_3.view(n, 1, h, w).cuda()
        merge_sem = torch.cat((merge_1, merge_2, merge_3), 1)            
        
        return merge_sem

    def forward(self):
        # stage1: sate rgb -> sate depth + sate label
        if self.is_train:
            stage1_input = {'sate_R': self.sate_real_R,
                    'sate_D': self.sate_real_D,
                    'sate_L': self.sate_real_L,                   
                    'img_id': self.img_id
            }
        else:
            stage1_input = {'sate_R': self.sate_real_R,               
                    'img_id': self.img_id
            }
        self.net_stage1.set_input(stage1_input)
        self.net_stage1.forward()
        self.sate_fake_DL = self.net_stage1.fake_DL
        self.sate_fake_L = self.sate_fake_DL[:,0:1,:,:]
        self.sate_fake_D = self.sate_fake_DL[:,1:2,:,:]

        # stage2: sate depth+label -> proj depth+label -> street sem
        if self.is_train:
            stage2_input = {'sate_R': self.sate_real_R,
                    'sate_D': self.sate_fake_D,
                    'sate_L': self.sate_fake_L,
                    'street_M': self.street_real_M,                    
                    'street_S': self.street_real_S,    
                    'ori': self.ori,                    
                    'img_id': self.img_id
            }
        else:
            stage2_input = {'sate_R': self.sate_real_R,
                    'sate_D': self.sate_fake_D,
                    'sate_L': self.sate_fake_L,
                    'ori': self.ori,                    
                    'img_id': self.img_id
            }
        self.net_stage2.set_input(stage2_input)
        self.net_stage2.forward()
        self.proj_fake_depth = self.net_stage2.proj_D
        self.street_fake_S = self.net_stage2.fake_S
        
        # stage3: proj depth + street sem -> street rgb     
        self.merge_S = self.cvt_geo2bicycle()
        if self.is_train:
            stage3_input = {'A': self.merge_S,
                    'B': self.street_real_R,
                    'A_paths': 'A',                   
                    'B_paths': 'B'
            }
        else:
            stage3_input = {'A': self.merge_S,
                    'B': self.merge_S,
                    'A_paths': 'A',                   
                    'B_paths': 'B'
            }
        
        self.net_stage3.set_input(stage3_input)
        if self.is_train:
            self.net_stage3.forward()
        else:
            self.net_stage3.test()
            self.street_fake_R = self.net_stage3.fake_B
            self.street_fake_R = F.interpolate(self.street_fake_R, (256, 512))            

    def optimize_parameters(self):
        self.forward()

        # optimize stage3
        self.net_stage3.optimize_parameters_only()

        # optimize stage2
        self.net_stage2.optimize_parameters_only()

        # optimize stage1
        self.net_stage1.optimize_parameters_only()

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.net_stage1.netG.load_state_dict(torch.load(self.save_dir +'/model_S1_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))       
            self.net_stage2.netG.load_state_dict(torch.load(self.save_dir +'/model_S2_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))   
            self.net_stage3.netG.load_state_dict(torch.load(self.save_dir +'/model_S3_G_'+str(epoch)+'.pth',
            map_location=lambda storage, loc: storage.cuda(0)))  
            self.net_stage3.netD.load_state_dict(torch.load(self.save_dir +'/model_S3_D_'+str(epoch)+'.pth',
            map_location=lambda storage, loc: storage.cuda(0)))  
            self.net_stage3.netD2.load_state_dict(torch.load(self.save_dir +'/model_S3_D2_'+str(epoch)+'.pth',
            map_location=lambda storage, loc: storage.cuda(0)))  
            self.net_stage3.netE.load_state_dict(torch.load(self.save_dir +'/model_S3_E_'+str(epoch)+'.pth',
            map_location=lambda storage, loc: storage.cuda(0)))                          
        else:
            self.net_stage1.netG.load_state_dict(torch.load(self.save_dir +'/model_S1_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))       
            self.net_stage2.netG.load_state_dict(torch.load(self.save_dir +'/model_S2_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))               
            
            self.net_stage3.netG.load_state_dict({k.replace('model.','module.model.'):v for k,v in torch.load(self.save_dir +'/model_S3_G_latest.pth',
            map_location=lambda storage, loc: storage.cuda(0)).items()})                       
            self.net_stage3.netD.load_state_dict({k.replace('model_','module.model_'):v for k,v in torch.load(self.save_dir +'/model_S3_D_latest.pth',
            map_location=lambda storage, loc: storage.cuda(0)).items()})            
            self.net_stage3.netD2.load_state_dict({k.replace('model_','module.model_'):v for k,v in torch.load(self.save_dir +'/model_S3_D2_latest.pth',
            map_location=lambda storage, loc: storage.cuda(0)).items()})             
            self.net_stage3.netE.load_state_dict({'module.'+k:v for k,v in torch.load(self.save_dir +'/model_S3_E_latest.pth',
            map_location=lambda storage, loc: storage.cuda(0)).items()})     


    def save_networks(self, epoch):
        torch.save(self.net_stage1.netG.state_dict(), self.save_dir +'/model_S1_G_'+str(epoch)+'.pt')
        torch.save(self.net_stage1.netG.state_dict(), self.save_dir +'/model_S1_G_latest.pt')

        torch.save(self.net_stage2.netG.state_dict(), self.save_dir +'/model_S2_G_'+str(epoch)+'.pt')
        torch.save(self.net_stage2.netG.state_dict(), self.save_dir +'/model_S2_G_latest.pt')

        torch.save(self.net_stage3.netG.state_dict(), self.save_dir +'/model_S3_G_'+str(epoch)+'.pt')
        torch.save(self.net_stage3.netG.state_dict(), self.save_dir +'/model_S3_G_latest.pt')

        torch.save(self.net_stage3.netD.state_dict(), self.save_dir +'/model_S3_D_'+str(epoch)+'.pt')
        torch.save(self.net_stage3.netD.state_dict(), self.save_dir +'/model_S3_D_latest.pt')

        torch.save(self.net_stage3.netD2.state_dict(), self.save_dir +'/model_S3_D2_'+str(epoch)+'.pt')
        torch.save(self.net_stage3.netD2.state_dict(), self.save_dir +'/model_S3_D2_latest.pt')        

        torch.save(self.net_stage3.netE.state_dict(), self.save_dir +'/model_S3_E_'+str(epoch)+'.pt')
        torch.save(self.net_stage3.netE.state_dict(), self.save_dir +'/model_S3_E_latest.pt')   
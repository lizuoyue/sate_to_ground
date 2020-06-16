import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, utils
import cv2
import numpy

import networks
from geo_process_layer import geo_projection
from utils import ImagePool

# stage 1: sate rgb -> depth + label
class R2DL(nn.Module):
    def name(self):
        return 'R2DL'

    def __init__(self, opt):
        super().__init__()
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')           

        # load/define networks
        #self.netG = networks.define_X(1, 3, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(3, 2, opt.ngf_S12, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        self.real_R = input['sate_R'].to(self.device)
        if self.is_train:
            self.real_D = input['sate_D'].to(self.device)
            self.real_L = input['sate_L'].to(self.device)
        self.img_id = input['img_id']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_DL = self.netG(self.real_R)
        self.fake_D = self.fake_DL[:,0:1,:,:]
        self.fake_L = self.fake_DL[:,1:2,:,:]

    def backward_G(self):
        # Second, G(A) = B
        self.loss_G_Loss1 = self.criterionL1(self.fake_D, self.real_D)
        self.loss_G_Loss2 = self.criterionL1(self.fake_L, self.real_L)

        self.loss_G = self.loss_G_Loss1 + self.loss_G_Loss2*0.5

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters_only(self):
        # update G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_networks(self, epoch):
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))       
        else:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))


# stage 2: projected depth+label->street semantic
class DL2S(nn.Module):
    def name(self):
        return 'DL2S'

    def __init__(self, opt):
        super().__init__()
        self.sate_gsd = opt.sate_gsd
        self.pano_size = opt.pano_size        
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')           
 
        self.netG = networks.define_G(5, 3, opt.ngf_S12, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)         

        if self.is_train:
            self.optimizers = []
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):        
        self.real_R = input['sate_R'].to(self.device)
        self.real_D = input['sate_D'].to(self.device)
        self.real_L = input['sate_L'].to(self.device)
        if self.is_train:            
            self.real_S = input['street_S'].to(self.device)
            self.real_M = input['street_M'].to(self.device)
        self.img_id = input['img_id']
        self.ori = input['ori']

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(numpy.concatenate([init_dim * numpy.arange(n_tile) + i for i in range(init_dim)])).cuda()
        return torch.index_select(a, dim, order_index)

    def forward(self):
        # step1: geo-projection
        self.proj_D, self.proj_L = geo_projection(self.real_D, self.real_L, self.ori, self.sate_gsd, self.pano_size, is_normalized=True)

        # step2: RDL -> S        
        self.resize_R = self.tile(self.real_R, 3, 2)
        self.fake_S = self.netG(torch.cat((self.resize_R, self.proj_D, self.proj_L), 1)) # 3, 1, 1

    def backward(self):
        # RDL-S loss
        self.dev = torch.abs(self.fake_S - self.real_S)
        self.loss_G_Loss = self.dev*0.1 + torch.mul(self.dev, self.real_M)    # weighted loss
        self.loss_G = torch.mean(self.loss_G_Loss)  

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.backward()

    def optimize_parameters_only(self):
        # update G
        self.backward()

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))                                              
        else:
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))         
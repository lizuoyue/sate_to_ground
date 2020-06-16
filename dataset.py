"""
UNet
opturations and data loading code for Kaggle Data Science Bowl 2018
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
from skimage.transform import resize
import matplotlib.pylab as plt
import re
import cv2


# Load Data RGB to Depth data
class MyDataLoader(Dataset):
    def __init__(self, root_dir, train=True, coarse=True):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        # get data
        self.img_id = []
        self.pos_id = []
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        name_rgb = '_sate_rgb_00.png'
        len_rgb = len(name_rgb)
        for img_name in self.img_all:
            print(img_name)
            if img_name[-len_rgb:] == name_rgb: 
                name = img_name[:-len_rgb]                
                self.img_id.append(name)
                self.pos_id.append('_00.png')

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):        
        dir_sate_rgb = self.root_dir + '/' + self.img_id[idx] + '_sate_rgb' + self.pos_id[idx]
        sate_rgb = io.imread(dir_sate_rgb).astype(np.uint8)              
        sate_rgb = transforms.ToTensor()(sate_rgb)
        sate_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_rgb).float()     
        #color_rgb = [[70,70,70], [70,130,180], [106,142,34], [243,36,232], [128,64,127]]         
        if self.train:
            # sate label
            dir_sate_label = self.root_dir + '/' + self.img_id[idx] + '_sate_label'+self.pos_id[idx]            
            sate_label = io.imread(dir_sate_label).astype(np.float)
            sate_label_single = sate_label[:,:,0]*255*255 + sate_label[:,:,1]*255 + sate_label[:,:,2]
            sate_label_single[sate_label_single==(70*255*255+70*255+70)] = 128
            sate_label_single[sate_label_single==(106*255*255+142*255+34)] = 64
            sate_label_single[sate_label_single==(128*255*255+64*255+127)] = 0
            sate_label[:,:,0] = sate_label_single
            sate_label[:,:,1] = sate_label_single
            sate_label[:,:,2] = sate_label_single   
            sate_label = transforms.ToTensor()(sate_label)             
            sate_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_label).float()            
            sate_label = sate_label[0:1,:,:]

            # sate depth
            dir_sate_depth = self.root_dir + '/' + self.img_id[idx] + '_sate_depth'+self.pos_id[idx]            
            sate_depth = io.imread(dir_sate_depth).astype(np.float)   
            sate_depth = transforms.ToTensor()(sate_depth)               
            sate_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_depth).float()            
            sate_depth = sate_label[0:1,:,:]

            # proj dis
            dir_proj_dis = self.root_dir + '/' + self.img_id[idx] + '_proj_dis'+self.pos_id[idx]
            proj_dis = io.imread(dir_proj_dis).astype(np.uint8)
            proj_dis = transforms.ToTensor()(proj_dis)
            proj_dis = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(proj_dis).float()
            proj_dis = proj_dis[0:1,:,:]

            # proj label
            dir_proj_label = self.root_dir + '/' + self.img_id[idx] + '_proj_label'+self.pos_id[idx]
            proj_label = io.imread(dir_proj_label).astype(np.float)
            proj_label_single = proj_label[:,:,0]*255*255 + proj_label[:,:,1]*255 + proj_label[:,:,2]
            proj_label_single[proj_label_single==(70*255*255+130*255+180)] = 255
            proj_label_single[proj_label_single==(70*255*255+70*255+70)] = 128
            proj_label_single[proj_label_single==(128*255*255+64*255+127)] = 0
            proj_label[:,:,0] = proj_label_single
            proj_label[:,:,1] = proj_label_single
            proj_label[:,:,2] = proj_label_single    
            proj_label = transforms.ToTensor()(proj_label)            
            proj_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(proj_label).float()            
            proj_label = proj_label[0:1,:,:]

            # proj mask          
            dir_str_mask = self.root_dir + '/' + self.img_id[idx] + '_proj_mask'+self.pos_id[idx]
            str_mask = io.imread(dir_str_mask).astype(np.int)
            str_mask = str_mask[:,:,0]
            str_mask = (str_mask>0).astype(np.int)           
            str_mask = torch.from_numpy(str_mask).type(torch.float) 

            # street label
            dir_str_sem = self.root_dir + '/' + self.img_id[idx] + '_street_label'+self.pos_id[idx]            
            str_sem = io.imread(dir_str_sem).astype(np.uint8)
            str_sem = transforms.ToTensor()(str_sem)
            str_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(str_sem).float()

            # street rgb
            dir_str_rgb = self.root_dir + '/' + self.img_id[idx] + 'img_street_rgb'+self.pos_id[idx]            
            str_rgb = io.imread(dir_str_rgb).astype(np.uint8)
            str_rgb = transforms.ToTensor()(str_rgb)
            str_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(str_rgb).float()

        iid = [m.start() for m in re.finditer(',', self.img_id[idx])]
        ori = float(self.img_id[idx][iid[1]+1:])/180*3.1415926545         

        # to tensor
        if self.train:
            return {'sate_R': sate_rgb,
                    'sate_D': sate_depth,
                    'sate_L': sate_label,
                    'proj_D': proj_dis,  
                    'proj_L': proj_label,                    
                    'street_S': str_sem,    
                    'street_M': str_mask,    
                    'street_R': str_rgb,    
                    'ori': ori,                    
                    'img_id': self.img_id[idx]}
        else:
            return {'sate_R': sate_rgb,
                    'ori': ori,                    
                    'img_id': self.img_id[idx]}      

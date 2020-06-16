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
from utils import Option
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import re


# Load Data RGB to Depth data
class R2DDataLoader(Dataset):
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
        for img_name in self.img_all:
            if img_name[-16:-7] == '_sate_rgb': 
                name = img_name[:-16]
                post_fix = img_name[-6:]
                self.img_id.append(root_dir+'/'+name)
                self.pos_id.append(post_fix)

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        dir_rgb = self.img_id[idx] + '_sate_rgb_'+self.pos_id[idx]
        img_rgb = io.imread(dir_rgb).astype(np.uint8)
        img_rgb = transforms.ToTensor()(img_rgb)
        img_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_rgb).float()

        if self.train:      
            dir_label = self.img_id[idx] + '_sate_label_'+self.pos_id[idx]
            dir_depth = self.img_id[idx] + '_sate_depth_'+self.pos_id[idx]

            img_label = io.imread(dir_label).astype(np.uint8)
            img_label[img_label==1]=0
            img_label = color.grey2rgb(img_label)
            #h, w = img_label.shape
            #img_label = np.reshape(img_label, (h,w,1))
            img_label = transforms.ToTensor()(img_label)

            img_depth = io.imread(dir_depth).astype(np.uint8)
            #img_depth = color.rgb2grey(img_depth)
            #h, w = img_depth.shape
            #img_depth = np.reshape(img_depth, (h,w,1))
            img_depth = transforms.ToTensor()(img_depth)
                
            img_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_label).float()
            img_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_depth).float()
            img_label = img_label[0:1,:,:]
            img_depth = img_depth[0:1,:,:]

        # to tensor
        if self.train:
            return {'R': img_rgb,
                    'D': img_depth,
                    'L': img_label,
                    'img_id': self.img_id[idx]}
        else:
            return {'R': img_rgb,
                    'img_id': self.img_id[idx]}            

# Load Data RGB to Depth data
class D2LDataLoader(Dataset):
    def __init__(self, root_dir, train=True, coarse=True, fine_tune_sidewalk=False):
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
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        for img_name in self.img_all:
            if img_name[-13:] == '_proj_sem.png':
                self.img_id.append(img_name[:-13])

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse
        self.fine_tune_sidewalk = fine_tune_sidewalk

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        dir_label = self.root_dir + '/' + self.img_id[idx] + '_proj_sem.png'
        dir_depth = self.root_dir + '/' + self.img_id[idx] + '_proj_depth.png'

        if self.train:
            dir_sem = self.root_dir + '/' + self.img_id[idx] + '_street_sem4.png'
            img_sem = io.imread(dir_sem).astype(np.uint8)
            img_sem = transforms.ToTensor()(img_sem)
            img_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_sem).float()

            dir_mask = self.root_dir + '/' + self.img_id[idx] + '.png'
            img_mask = io.imread(dir_mask).astype(np.uint8)
            img_mask = color.rgb2grey(img_mask)
            h, w = img_mask.shape
            img_mask = np.reshape(img_mask, (h,w,1))
            img_mask = transforms.ToTensor()(img_mask)

            if self.fine_tune_sidewalk:
                dir_mask_swalk = self.root_dir + '/' + self.img_id[idx] + '_mask_sidewalk.png'
                img_mask_swalk = io.imread(dir_mask_swalk).astype(np.uint8)
                img_mask_swalk = color.rgb2grey(img_mask_swalk)
                h, w = img_mask_swalk.shape
                img_mask_swalk = np.reshape(img_mask_swalk, (h,w,1))
                img_mask_swalk = transforms.ToTensor()(img_mask_swalk)

        img_label = io.imread(dir_label).astype(np.uint8)
        img_label = color.rgb2grey(img_label)
        h, w = img_label.shape
        img_label = np.reshape(img_label, (h,w,1))
        img_label = transforms.ToTensor()(img_label)

        img_depth = io.imread(dir_depth).astype(np.uint8)
        img_depth = color.rgb2grey(img_depth)
        h, w = img_depth.shape
        img_depth = np.reshape(img_depth, (h,w,1))
        img_depth = transforms.ToTensor()(img_depth)
        
        img_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_label).float()
        img_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_depth).float()

        #print(self.img_id[idx])

        # to tensor
        if self.train and self.fine_tune_sidewalk:
            return {'S': img_sem,
                    'D': img_depth,
                    'M': img_mask,
                    'W': img_mask_swalk,
                    'L': img_label,
                    'img_id': self.img_id[idx]}     
        if self.train:
            return {'S': img_sem,
                    'D': img_depth,
                    'M': img_mask,
                    'L': img_label,
                    'img_id': self.img_id[idx]}                
        else:
            return {'D': img_depth,
                    'L': img_label,
                    'img_id': self.img_id[idx]}                    



# Load Data RGB to Depth data
class L2RDataLoader(Dataset):
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
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        for img_name in self.img_all:
            if img_name[-13:] == '_sate_rgb.png':
                self.img_id.append(img_name[:-13])

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if self.train:
            dir_rgb = self.root_dir + '/' + self.img_id[idx] + '_street_rgb.png'
            img_rgb = io.imread(dir_rgb).astype(np.uint8)
            img_rgb = transforms.ToTensor()(img_rgb)
            img_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_rgb).float()         

        #dir_label = self.root_dir + '/' + self.img_id[idx] + '_street_sem3.png'
        #dir_label = self.root_dir + '/' + self.img_id[idx] + '_pred_sem2.png'
        dir_label = self.root_dir + '/' + self.img_id[idx] + '_pred_sem_dll.png'
        dir_proj_rgb = self.root_dir + '/' + self.img_id[idx] + '_pred_sem_dll.png'      

        img_label = io.imread(dir_label).astype(np.uint8)
        img_label = transforms.ToTensor()(img_label)

        if 0:
            img_proj_rgb = io.imread(dir_proj_rgb).astype(np.uint8)
            img_proj_rgb = color.rgb2grey(img_proj_rgb)
            h, w = img_proj_rgb.shape
            img_proj_rgb = np.reshape(img_proj_rgb, (h,w,1))
            img_proj_rgb = transforms.ToTensor()(img_proj_rgb)
        else:
            img_proj_rgb = io.imread(dir_proj_rgb).astype(np.uint8)
            img_proj_rgb = transforms.ToTensor()(img_proj_rgb)

        img_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_label).float()
        img_proj_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_proj_rgb).float()

        # to tensor
        if self.train:
            return {'R': img_rgb,
                    'L': img_label,
                    'Proj_R': img_proj_rgb,
                    'img_id': self.img_id[idx]}
        else:
            return {'L': img_label,
                    'Proj_R': img_proj_rgb,
                    'img_id': self.img_id[idx]}            


# Load Data RGB to Depth data
class DLRDataLoader(Dataset):
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
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        for img_name in self.img_all:
            if img_name[-13:] == '_proj_sem.png':
                self.img_id.append(img_name[:-13])

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        dir_label = self.root_dir + '/' + self.img_id[idx] + '_proj_sem.png'
        dir_depth = self.root_dir + '/' + self.img_id[idx] + '_proj_depth.png'
        dir_rgb = self.root_dir + '/' + self.img_id[idx] + '_proj_rgb.png'

        if self.train:
            dir_sem = self.root_dir + '/' + self.img_id[idx] + '_street_sem2.png'
            img_sem = io.imread(dir_sem).astype(np.uint8)
            # img_sem = color.rgb2grey(img_sem)
            # h, w = img_sem.shape
            # img_sem = np.reshape(img_sem, (h,w,1))
            img_sem = transforms.ToTensor()(img_sem)
            img_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_sem).float()

            dir_gt_rgb = self.root_dir + '/' + self.img_id[idx] + '_street_rgb.png'
            img_gt_rgb = io.imread(dir_gt_rgb).astype(np.uint8)          
            img_gt_rgb = transforms.ToTensor()(img_gt_rgb)
            img_gt_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_gt_rgb).float()

            # mask
            dir_mask = self.root_dir + '/' + self.img_id[idx] + '.png'
            img_mask = io.imread(dir_mask).astype(np.uint8)
            img_mask = color.rgb2grey(img_mask)
            h, w = img_mask.shape
            img_mask = np.reshape(img_mask, (h,w,1))
            img_mask = transforms.ToTensor()(img_mask)

            dir_mask_swalk = self.root_dir + '/' + self.img_id[idx] + '_mask_sidewalk.png'
            img_mask_swalk = io.imread(dir_mask_swalk).astype(np.uint8)
            img_mask_swalk = color.rgb2grey(img_mask_swalk)
            h, w = img_mask_swalk.shape
            img_mask_swalk = np.reshape(img_mask_swalk, (h,w,1))
            img_mask_swalk = transforms.ToTensor()(img_mask_swalk)

            img_mask = img_mask - img_mask_swalk

        img_label = io.imread(dir_label).astype(np.uint8)
        img_label = color.rgb2grey(img_label)
        h, w = img_label.shape
        img_label = np.reshape(img_label, (h,w,1))
        img_label = transforms.ToTensor()(img_label)

        img_depth = io.imread(dir_depth).astype(np.uint8)
        img_depth = color.rgb2grey(img_depth)
        h, w = img_depth.shape
        img_depth = np.reshape(img_depth, (h,w,1))
        img_depth = transforms.ToTensor()(img_depth)

        img_rgb = io.imread(dir_rgb).astype(np.uint8)          
        img_rgb = transforms.ToTensor()(img_rgb)

        img_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_rgb).float()
        img_label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_label).float()
        img_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_depth).float()

        # to tensor
        if self.train:
            return {'proj_D': img_depth,
                    'proj_R': img_rgb,
                    'proj_L': img_label,
                    'real_R': img_gt_rgb,
                    'real_L': img_sem,
                    'M': img_mask,
                    'W': img_mask_swalk,
                    'img_id': self.img_id[idx]}     
        else:
            return {'proj_D': img_depth,
                    'proj_R': img_rgb,
                    'proj_L': img_label,
                    'img_id': self.img_id[idx]}                 



class RDLRDataLoader(Dataset):
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
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        for img_name in self.img_all:
            if img_name[-16:] == '_sate_rgb_05.png':
                self.img_id.append(img_name[:-16])

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        dir_sate_rgb = self.root_dir + '/' + self.img_id[idx] + '_sate_rgb_05.png'
        sate_rgb = io.imread(dir_sate_rgb).astype(np.uint8)          
        sate_rgb = transforms.ToTensor()(sate_rgb)
        sate_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_rgb).float()

        if self.train:
            dir_sate_sem = self.root_dir + '/' + self.img_id[idx] + '_sate_sem.png'
            sate_sem = io.imread(dir_sate_sem).astype(np.uint8)
            sate_sem = color.rgb2grey(sate_sem)
            h, w = sate_sem.shape
            sate_sem = np.reshape(sate_sem, (h,w,1))            
            sate_sem = transforms.ToTensor()(sate_sem)
            sate_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_sem).float()

            dir_sate_depth = self.root_dir + '/' + self.img_id[idx] + '_sate_depth.png'
            sate_depth = io.imread(dir_sate_depth).astype(np.uint8)
            sate_depth = color.rgb2grey(sate_depth)
            h, w = sate_depth.shape
            sate_depth = np.reshape(sate_depth, (h,w,1))            
            sate_depth = transforms.ToTensor()(sate_depth)
            sate_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_depth).float()

            dir_street_sem = self.root_dir + '/' + self.img_id[idx] + '_street_sem2.png'
            street_sem = io.imread(dir_street_sem).astype(np.uint8)
            street_sem = transforms.ToTensor()(street_sem)
            street_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(street_sem).float()

            dir_street_rgb = self.root_dir + '/' + self.img_id[idx] + '_street_rgb.png'
            street_rgb = io.imread(dir_street_rgb).astype(np.uint8)          
            street_rgb = transforms.ToTensor()(street_rgb)
            street_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(street_rgb).float()

        iid = [m.start() for m in re.finditer(',', self.img_id[idx])]
        ori = float(self.img_id[idx][iid[1]+1:])/180*3.1415926545

        # to tensor
        if self.train:
            return {'sate_R': sate_rgb,
                    'sate_D': sate_depth,
                    'sate_L': sate_sem,
                    'street_R': street_rgb,
                    'street_L': street_sem,
                    'ori': ori,
                    'img_id': self.img_id[idx]}     
        else:
            return {'sate_R': sate_rgb,
                    'ori': ori,
                    'img_id': self.img_id[idx]}    



class DLLDataLoader(Dataset):
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
        self.img_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir,f))]
        for img_name in self.img_all:
            if img_name[-13:] == '_sate_rgb.png':
                self.img_id.append(img_name[:-13])

        self.root_dir = root_dir
        self.train = train
        self.coarse = coarse

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if not self.train:
            dir_sate_sem = self.root_dir + '/' + self.img_id[idx] + '_pred_sate_label.png'
        else:
            dir_sate_sem = self.root_dir + '/' + self.img_id[idx] + '_sate_sem.png'
        sate_sem = io.imread(dir_sate_sem).astype(np.uint8)
        sate_sem = color.rgb2grey(sate_sem)
        h, w = sate_sem.shape
        sate_sem = np.reshape(sate_sem, (h,w,1))            
        sate_sem = transforms.ToTensor()(sate_sem)
        sate_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_sem).float()

        if not self.train:
            dir_sate_depth = self.root_dir + '/' + self.img_id[idx] + '_pred_depth_finetune.png'
        else:
            dir_sate_depth = self.root_dir + '/' + self.img_id[idx] + '_sate_depth.png'
        sate_depth = io.imread(dir_sate_depth).astype(np.uint8)
        sate_depth = color.rgb2grey(sate_depth)
        h, w = sate_depth.shape
        sate_depth = np.reshape(sate_depth, (h,w,1))            
        sate_depth = transforms.ToTensor()(sate_depth)
        sate_depth = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sate_depth).float()

        if self.train:
            dir_street_sem = self.root_dir + '/' + self.img_id[idx] + '_street_sem2.png'
            street_sem = io.imread(dir_street_sem).astype(np.uint8)
            street_sem = resize(street_sem, (256, 256), anti_aliasing=True)
            street_sem = transforms.ToTensor()(street_sem)
            street_sem = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(street_sem).float()

        iid = [m.start() for m in re.finditer(',', self.img_id[idx])]
        ori = float(self.img_id[idx][iid[1]+1:])/180*3.1415926545

        # to tensor
        if self.train:
            return {'sate_D': sate_depth,
                    'sate_L': sate_sem,
                    'street_L': street_sem,
                    'ori': ori,
                    'img_id': self.img_id[idx]}     
        else:
            return {'sate_D': sate_depth,
                    'sate_L': sate_sem,
                    'ori': ori,
                    'img_id': self.img_id[idx]} 
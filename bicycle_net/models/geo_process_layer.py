import os
import sys
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils

from skimage import io, color
from os import listdir
from os.path import isfile, join
import re

def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)

    grid = torch.tensor([xy]).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid

def depth2voxel(img_depth, img_sem, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    site_z = img_depth[:, 0, int(h/2), int(w/2)] + 3.0
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize).cuda()

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.expand(n, gsize, gsize, 2).cuda()
    grid_depth = torch.nn.functional.grid_sample(img_depth.cuda(), grid_mask)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

    # semantic voxel
    grid_s = torch.nn.functional.grid_sample(img_sem.view(n, 1, h, w), grid_mask)
    voxel_s = grid_s.expand(n, gsize, gsize, gsize).cuda()

    k = int(1)
    voxel_s = grid_s.expand(n, gsize/2+k, gsize, gsize)
    gound_s = torch.zeros([n,gsize/2-k,gsize, gsize], dtype=torch.float) # set as ground
    voxel_s = torch.cat((gound_s, voxel_s.cpu()), 1).cuda()

    # occupancy voxel
    voxel_grid = torch.arange(-gsize/2, gsize/2, 1).float()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize).cuda()
    voxel_ocupy = torch.ge(voxel_depth, voxel_grid).float().cpu()
    voxel_ocupy[:,gsize-1,:,:] = 0
    voxel_ocupy = voxel_ocupy.cuda()

    # distance voxel
    voxel_dx = grid_mask[0,:,:,0].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dx = voxel_dx.cuda().mul(voxel_ocupy)
    voxel_dy = grid_mask[0,:,:,1].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dy = voxel_dy.cuda().mul(voxel_ocupy)

    voxel_delta = torch.argmin(voxel_ocupy, 1).mul(-1).view(n, 1, gsize, gsize).expand(n, gsize, gsize, gsize).float().cuda()
    voxel_delta = voxel_delta + torch.arange(0, gsize, 1).float().view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize).cuda()
    grid_depth = torch.min(grid_depth, site_z.add(float(gsize/2.0)).view(n,1,1,1).expand(n,1,gsize,gsize).cuda())
    voxel_dz = grid_depth.expand(n, gsize, gsize, gsize) 
    voxel_dz = voxel_dz + voxel_delta - voxel_sitez
    voxel_dz = voxel_dz.cuda().mul(voxel_ocupy)
 
    voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    voxel_dis = voxel_dis.add(0.01)   # avoid 1/0 = nan
    voxel_dis = torch.sqrt(voxel_dis) - voxel_ocupy.add(-1.0).mul(float(gsize)*0.9)

    # facade:128, tree:64, ground:0, sky:255
    voxel_s = voxel_s.mul(voxel_ocupy) - voxel_ocupy.add(-1.0).mul(255) 

    return voxel_dis, voxel_s

def voxel2pano(voxel_dis, voxel_s, ori, size_pano, is_normalized=True):
    PI = 3.1415926535
    r, c = [size_pano[0], size_pano[1]]
    n, s, t, tt = voxel_dis.size()
    k = int(s/2)

    # rays
    ori = ori.view(n, 1).expand(n, c).float()
    x = torch.arange(0, c, 1).float().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().view(1, r).expand(n, r)
    lon = x * 2 * PI/c + ori - PI
    lat = PI/2.0 - y * PI/r
    sin_lat = torch.sin(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    vx =  cos_lat.mul(sin_lon)
    vy = -cos_lat.mul(cos_lon)
    vz =  sin_lat
    vx = vx.expand(n, k, r, c)
    vy = vy.expand(n, k, r, c)
    vz = vz.expand(n, k, r, c)

    #
    voxel_dis = voxel_dis.contiguous().view(1, n*s*s*s)    
    voxel_s = voxel_s.contiguous().view(1, n*s*s*s)

    # sample voxels along pano-rays
    d_samples = torch.arange(0, float(k), 1).view(1, k, 1, 1).expand(n, k, r, c)
    samples_x = vx.mul(d_samples).add(k).long()
    samples_y = vy.mul(d_samples).add(k).long()
    samples_z = vz.mul(d_samples).add(k).long()
    samples_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, k, r, c).long()
    samples_indices = samples_n.mul(s*s*s).add(samples_z.mul(s*s)).add(samples_y.mul(s)).add(samples_x)
    samples_indices = samples_indices.view(1, n*k*r*c)
    samples_indices = samples_indices[0,:].cuda()

    # get depth pano
    samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
    samples_depth = samples_depth.view(n, k, r, c)
    min_depth = torch.min(samples_depth, 1)
    pano_depth = min_depth[0]
    pano_depth = pano_depth.view(n, 1, r, c)

    # get sem pano
    idx_z = min_depth[1].cpu().long()
    idx_y = torch.arange(0, r, 1).view(1, r, 1).expand(n, r, c).long()
    idx_x = torch.arange(0, c, 1).view(1, 1, c).expand(n, r, c).long()
    idx_n = torch.arange(0, n, 1).view(n, 1, 1).expand(n, r, c).long()
    idx = idx_n.mul(k*r*c).add(idx_z.mul(r*c)).add(idx_y.mul(c)).add(idx_x).view(1, n*r*c).cuda()
        
    samples_s = torch.index_select(voxel_s, 1, samples_indices)
    pano_sem = torch.index_select(samples_s, 1, idx[0,:]).view(n,1,r,c).float()

    return pano_depth, pano_sem

def street2sate(pano_depth, pano_rgb, ori, sate_gsd, sate_size):    
    pano_depth = pano_depth.cpu()
    pano_rgb = pano_rgb.cpu()
    PI = 3.1415926535    
    n, t, r, c = pano_rgb.size()
    s = sate_size

    # rays
    ori = ori.view(n, 1).expand(n, c).float()
    x = torch.arange(0, c, 1).float().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().view(1, r).expand(n, r)
    lon = x * 2 * PI/c + ori - PI
    lat = PI/2.0 - y * PI/r
    cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    vx =  cos_lat.mul(sin_lon)
    vy = -cos_lat.mul(cos_lon)
    
    # XY coordinates
    depth_valid = pano_depth<=115.0
    pano_depth = (pano_depth.float()) * (depth_valid.float())
    Xm = pano_depth.float().mul(vx).add(s/2*sate_gsd)   # in meter
    Ym = pano_depth.float().mul(vy).add(s/2*sate_gsd)   # in meter
    xp = torch.floor(Xm.div(sate_gsd))  # in pixel    
    yp = torch.floor(Ym.div(sate_gsd))  # in pixel
    sate_valid = (xp>0)*(xp<s)*(yp>0)*(yp<s)
    sate_x = (xp * sate_valid.float()).long()
    sate_y = (yp * sate_valid.float()).long()    
    sate_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, 1, r, c).long()    
    sate_idx = sate_n*s*s + sate_y*s + sate_x
    sate_idx = sate_idx.contiguous().view(n*r*c)    

    # r    
    street_r = pano_rgb[:,0,:,:].contiguous().view(n*r*c)   
    sate_r = torch.zeros([n,1,s,s])
    sate_r = sate_r.contiguous().view(n*s*s)    
    sate_r[sate_idx] = street_r
    sate_r = sate_r.contiguous().view(n,1,s,s) 

    # g   
    street_g = pano_rgb[:,1,:,:].contiguous().view(n*r*c)   
    sate_g = torch.zeros([n,1,s,s])
    sate_g = sate_g.contiguous().view(n*s*s)    
    sate_g[sate_idx] = street_g
    sate_g = sate_g.contiguous().view(n,1,s,s) 

    # b    
    street_b = pano_rgb[:,2,:,:].contiguous().view(n*r*c)   
    sate_b = torch.zeros([n,1,s,s])
    sate_b = sate_b.contiguous().view(n*s*s)    
    sate_b[sate_idx] = street_b   
    sate_b = sate_b.contiguous().view(n,1,s,s) 

    # rgb
    sate_rgb = torch.cat((sate_r, sate_g, sate_b), 1)

    return sate_rgb

def geo_projection(sate_depth, sate_sem, orientations, sate_gsd, pano_size, is_normalized=True):
    # recover the real depth
    if is_normalized:
        sate_depth = sate_depth.mul(0.5).add(0.5).mul(255)
        sate_sem = sate_sem.mul(0.5).add(0.5).mul(255)
    
    # step1: depth to voxel
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_d, voxel_s = depth2voxel(sate_depth, sate_sem, gsize)

    # step2: voxel to panorama
    pano_depth, pano_sem = voxel2pano(voxel_d, voxel_s, orientations, pano_size, False) 

    #step3: change pixel values of semantic panorama
    pano_sem = pano_sem.mul(1.0/255.0)
    pano_sem = pano_sem.add(-0.5).div(0.5)

    pano_depth = pano_depth.mul(1.0/116.0)
    pano_depth = pano_depth.add(-0.5).div(0.5)

    return pano_depth, pano_sem 

def geo_reprojection(pano_depth, pano_rgb, orientations, sate_gsd, sate_size, is_normalized=True):
    # print(pano_depth.shape)
    # print(pano_rgb.shape)
    # print(orientations.shape)
    # recover the real depth and color
    if is_normalized:
        pano_depth = pano_depth.mul(0.5).add(0.5).mul(116.0)
        pano_rgb = pano_rgb.mul(0.5).add(0.5).mul(255.0)
    
    sate_rgb = street2sate(pano_depth, pano_rgb, orientations, sate_gsd, sate_size)
    sate_rgb = sate_rgb.div(255.0).add(-0.5).div(0.5)
    #torchvision.utils.save_image(sate_rgb[0,:,:,:]*0.5+0.5, "F:\\sate.png")

    return sate_rgb

if __name__ == "__main__":
    # read in data
    root_dir = "C:/Users/lu.2037/Desktop/ttt"
    name = "-0.1172482702384059,17.13496208190918,51.5037964338242"
    file_rgb = root_dir + "/" + name + "_pred_rgb.png"
    sate_rgb = io.imread(file_rgb).astype(np.uint8) 
    sate_rgb = torch.from_numpy(sate_rgb.transpose((2, 0, 1))).float()            
    c, h, w = sate_rgb.size()
    sate_rgb = sate_rgb.view(1, c, h, w)
    sate_rgb = torch.autograd.Variable(sate_rgb,requires_grad=True)

    file_depth = root_dir + "/" + name + "_proj_dis.png"
    sate_depth = io.imread(file_depth).astype(np.uint8)
    sate_depth = color.rgb2grey(sate_depth)
    h, w = sate_depth.shape
    sate_depth = np.reshape(sate_depth, (h,w,1))            
    sate_depth = transforms.ToTensor()(sate_depth)
    c, h, w = sate_depth.size()
    sate_depth = sate_depth.view(1, c, h, w).mul(116.0)

    orientations = torch.tensor(51.5037964338242/180.0*3.1415926)
    ttt = geo_reprojection(sate_depth, sate_rgb, orientations, 0.5, 256, False)
    vv = torch.sum(ttt)
    vv.backward()
    print(sate_rgb.grad[0,0,128,:])
    print(sate_rgb.grad[0,1,128,:])


    
import os
import sys
import numpy as np
import torch
import torchvision
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

def depth2voxel(img_depth, img_rgb, gsize):
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

    k = int(1)

    # rgb voxel
    r = img_rgb[:,2,:,:].contiguous().view(n, 1, h, w)
    voxel_r = torch.nn.functional.grid_sample(r.cuda(), grid_mask).expand(n, gsize, gsize, gsize)
    g = img_rgb[:,1,:,:].contiguous().view(n, 1, h, w)
    voxel_g = torch.nn.functional.grid_sample(g.cuda(), grid_mask).expand(n, gsize, gsize, gsize)
    b = img_rgb[:,0,:,:].contiguous().view(n, 1, h, w)
    voxel_b = torch.nn.functional.grid_sample(b.cuda(), grid_mask).expand(n, gsize, gsize, gsize)    

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
    voxel_r = voxel_r.mul(voxel_ocupy) - voxel_ocupy.add(-1.0).mul(70) 
    voxel_g = voxel_g.mul(voxel_ocupy) - voxel_ocupy.add(-1.0).mul(130) 
    voxel_b = voxel_b.mul(voxel_ocupy) - voxel_ocupy.add(-1.0).mul(180) 

    return voxel_dis, voxel_r, voxel_g, voxel_b

def voxel2pano(voxel_dis, voxel_r, voxel_g, voxel_b, ori, size_pano, is_normalized=True):
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
    voxel_r = voxel_r.contiguous().view(1, n*s*s*s)
    voxel_g = voxel_g.contiguous().view(1, n*s*s*s)
    voxel_b = voxel_b.contiguous().view(1, n*s*s*s)

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

    samples_r = torch.index_select(voxel_r, 1, samples_indices)
    samples_g = torch.index_select(voxel_g, 1, samples_indices)
    samples_b = torch.index_select(voxel_b, 1, samples_indices)
    pano_r = torch.index_select(samples_r, 1, idx[0,:]).view(n,1,r,c).float()
    pano_g = torch.index_select(samples_g, 1, idx[0,:]).view(n,1,r,c).float()
    pano_b = torch.index_select(samples_b, 1, idx[0,:]).view(n,1,r,c).float()
    pano_rgb = torch.cat((pano_r, pano_g, pano_b), 1)

    return pano_depth, pano_rgb

def geo_projection(sate_depth, sate_rgb, orientations, sate_gsd, pano_size, is_normalized=True):
    # recover the real depth
    if is_normalized:
        sate_depth = sate_depth.mul(0.5).add(0.5).mul(255)        
        sate_rgb = sate_rgb.mul(0.5).add(0.5).mul(255)
    
    # step1: depth to voxel
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_d, voxel_r, voxel_g, voxel_b = depth2voxel(sate_depth, sate_rgb, gsize)

    # step2: voxel to panorama
    pano_depth, pano_rgb = voxel2pano(voxel_d, voxel_r, voxel_g, voxel_b, orientations, pano_size, False) 

    #step3: change pixel values of semantic panorama
    pano_depth = pano_depth.mul(1.0/116.0)
    pano_depth = pano_depth.add(-0.5).div(0.5)

    pano_rgb = pano_rgb.mul(1.0/255.0)
    pano_rgb = pano_rgb.add(-0.5).div(0.5)

    return pano_depth, pano_rgb


if __name__ == "__main__":
    pass
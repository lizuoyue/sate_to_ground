import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils

from skimage import io, color
from skimage.filters import roberts, sobel_h, sobel_v
from skimage.measure import compare_ssim as ssim

from os import listdir
from os.path import isfile, join
import re
import cv2

def PSNR(true_frame, pred_frame):
    eps = 0.0001
    prediction_error = 0.0
    [h,w,c] = true_frame.shape
    dev_frame = pred_frame-true_frame
    dev_frame = np.multiply(dev_frame,dev_frame)
    prediction_error = np.sum(dev_frame)
    prediction_error = 128*128*prediction_error/(h*w*c)
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error


def SSIM(img1, img2):
    [h,w,c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:,:,0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:,:,0]
    score = ssim(img1, img2)
    return score


def L1difference(img_true,img_pred):
    [h,w] = img_true.shape
    true_gx = sobel_h(img_true)/4.0
    true_gy = sobel_v(img_true)/4.0
    pred_gx = sobel_h(img_pred)/4.0
    pred_gy = sobel_v(img_pred)/4.0
    dx = np.abs(true_gx-pred_gx)
    dy = np.abs(true_gy-pred_gy)
    prediction_error = np.sum(dx+dy)
    prediction_error=128*128*prediction_error/(h*w)
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((255*255)/ eps)/np.log(10)
    return prediction_error

if __name__ == "__main__":
    fold_pred = 'F:/DevelopCenter/papers/ICCV2019/materials/evaluation/p2p'
    fold_gt = 'F:/DevelopCenter/papers/ICCV2019/materials/evaluation/gt'

    #get images    
    PSNR_score_avg = 0
    SSIM_score_avg = 0
    L1_score_avg = 0
    file_list = [f for f in os.listdir(fold_pred) if os.path.isfile(os.path.join(fold_pred,f))]
    for f in file_list:        
        img_pred = cv2.imread(fold_pred+'/'+f).astype(np.float)/255.0
        img_gt = cv2.imread(fold_gt+'/'+f).astype(np.float)/255.0
        img_pred_gray = cv2.imread(fold_pred+'/'+f,0).astype(np.float)/255.0
        img_gt_gray = cv2.imread(fold_gt+'/'+f,0).astype(np.float)/255.0

        PSNR_score = PSNR(img_gt, img_pred)
        SSIM_score = SSIM(img_gt, img_pred)
        L1_score = L1difference(img_gt_gray, img_pred_gray)
        PSNR_score_avg = PSNR_score_avg + PSNR_score
        SSIM_score_avg = SSIM_score_avg + SSIM_score
        L1_score_avg = L1_score_avg + L1_score
    PSNR_score_avg = PSNR_score_avg/len(file_list)
    SSIM_score_avg = SSIM_score_avg/len(file_list)
    L1_score_avg = L1_score_avg/len(file_list)
    print(SSIM_score_avg)
    print(PSNR_score_avg)    
    print(L1_score_avg)    
    
    
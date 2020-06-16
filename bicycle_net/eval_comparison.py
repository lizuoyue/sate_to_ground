import tensorflow as tf
import numpy as np
import glob
from PIL import Image

def readImageList(path):
	li = sorted(glob.glob(path))
	return [np.array(Image.open(item).resize((512, 256))) for item in li]

im1 = tf.placeholder(tf.uint8, [None, None, None, None])
im2 = tf.placeholder(tf.uint8, [None, None, None, None])
psnr = tf.image.psnr(im1, im2, max_val=255)
ssim = tf.image.ssim(im1, im2, max_val=255)
sess = tf.Session()

xh1  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_init.png')
xh2  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_dll.png')
xh3  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_finetune.png')
cvpr = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_2018.png')
p2p  = readImageList('/home/zoli/xiaohu_new_data/comparison/*_pred_rgb_p2p.png')
gt   = readImageList('/home/zoli/xiaohu_new_data/comparison/*_street_pano.png')

for item in [xh1, xh2, xh3, cvpr, p2p]:
	psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gt, im2: item})
	print('PSNR: ', psnr_val.mean())
	print('SSIM: ', ssim_val.mean())
	print()

# PSNR:  13.251899
# SSIM:  0.28161803

# PSNR:  13.258847
# SSIM:  0.24283297

# PSNR:  13.6376
# SSIM:  0.31324238

# PSNR:  13.744952
# SSIM:  0.316897

# PSNR:  13.366726
# SSIM:  0.29660797

p2p  = readImageList('/home/zoli/xiaohu_new_data/comp_small/p2p/*.png')
xh   = readImageList('/home/zoli/xiaohu_new_data/comp_small/mine/*.png')
gt   = readImageList('/home/zoli/xiaohu_new_data/comp_small/gt/*.png')
cvpr = readImageList('/home/zoli/xiaohu_new_data/comp_small/cvpr2018/*.png')

for item in [xh, cvpr, p2p]:
	psnr_val, ssim_val = sess.run([psnr, ssim], feed_dict={im1: gt, im2: item})
	print('PSNR: ', psnr_val.mean())
	print('SSIM: ', ssim_val.mean())
	print()

# PSNR:  13.760339
# SSIM:  0.3324559

# PSNR:  14.285599
# SSIM:  0.35699907

# PSNR:  13.600463
# SSIM:  0.3142436


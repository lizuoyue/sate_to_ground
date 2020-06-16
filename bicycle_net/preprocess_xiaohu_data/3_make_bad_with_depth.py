from PIL import Image
import PIL
import glob
import numpy as np
import os
import tqdm

two_dim = np.array([
	[255, 255],
	[  0, 127],
	[127,   0],
	[  0, 255],
	[255,   0],
])

target = '../datasets/L2R_bad_with_depth/'
os.makedirs(target, exist_ok = True)
for mode in ['train', 'val']:
	with open(mode + '.txt', 'r') as f:
		lines = [line.strip() for line in f.readlines()]
	os.makedirs(target + mode, exist_ok = True)

	sem_path = '/home/zoli/xiaohu_new_data/predict_of_train/%s_pred_sem_label.png'
	rgb_path = '/home/zoli/xiaohu_new_data/train2/%s_street_rgb.png'
	dep_path = '/home/zoli/xiaohu_new_data/train2/%s_proj_dis.png'

	for line in tqdm.tqdm(lines):
		sem = np.array(Image.open(sem_path % line).resize((512, 256), PIL.Image.BILINEAR))
		dep = np.array(Image.open(dep_path % line).convert('L').resize((512, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(rgb_path % line).resize((512, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		info[..., 2] = dep
		for i in range(5):
			info[sem == i, :2] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/%s.png' % line 
		Image.fromarray(bi).save(target + mode + basename)

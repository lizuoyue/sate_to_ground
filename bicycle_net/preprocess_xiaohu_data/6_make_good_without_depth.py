from PIL import Image
import PIL
import glob
import numpy as np
import os
import tqdm

two_dim = np.array([
	[255, 255, 0],
	[  0, 127, 0],
	[127,   0, 0],
	[  0, 255, 0],
	[255,   0, 0],
])

target = '../datasets/L2R_good_without_depth/'
os.makedirs(target, exist_ok = True)
for mode in ['train', 'val']:
	with open(mode + '.txt', 'r') as f:
		lines = [line.strip() for line in f.readlines()]
	os.makedirs(target + mode, exist_ok = True)

	sem_path = '/home/zoli/xiaohu_new_data/train2/%s_street_label_1.png'
	rgb_path = '/home/zoli/xiaohu_new_data/train2/%s_street_rgb.png'

	for line in tqdm.tqdm(lines):
		sem = np.array(Image.open(sem_path % line).resize((512, 256), PIL.Image.BILINEAR))
		rgb = np.array(Image.open(rgb_path % line).resize((512, 256), PIL.Image.BILINEAR))
		info = rgb.copy()
		for i in range(5):
			info[sem == i] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/%s.png' % line 
		Image.fromarray(bi).save(target + mode + basename)

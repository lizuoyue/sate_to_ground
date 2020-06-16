import os
import glob
import random

dataset_dir = '/home/zoli/xiaohu_new_data/train2'
files = glob.glob(dataset_dir + '/*_sate_label.png')
names = [os.path.basename(item).replace('_sate_label.png', '') for item in files]
random.seed(7)
random.shuffle(names)
with open('val.txt', 'w') as f:
	for name in names[:64]:
		f.write('%s\n' % name)
with open('train.txt', 'w') as f:
	for name in names[64:]:
		f.write('%s\n' % name)

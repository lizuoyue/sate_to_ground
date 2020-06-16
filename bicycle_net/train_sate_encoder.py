import torch
import torchvision
import random
import numpy as np
from models import networks
from PIL import Image

class option(object):
	def __init__(self):
		self.output_nc = 3
		self.nz = 32
		self.nef = 96
		self.netE = 'resnet_256'
		self.norm = 'instance'
		self.nl = 'relu'
		self.init_type = 'xavier'
		self.init_gain = 0.02

		self.gpu_ids = []
		self.use_vae = True

		self.lr = 0.0002
		self.beta1 = 0.5
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		self.batch_size = 4
		self.epoch = 5
		return

if __name__=='__main__':
	opt = option()
	netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
							init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, vaeLike=opt.use_vae)
	criterionL1 = torch.nn.L1Loss()
	optimizer = torch.optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	d = np.load('encoded_z_aug_expand.npy', allow_pickle=True).item()
	names = sorted(list(d.keys()))
	vectors = [d[name] for name in names]
	paths = ['/home/zoli/xiaohu_new_data/train_augment/train_0/%s_sate_rgb_00.png' % name.replace('_00', '') for name in names]
	# paths = ['./test_sate/%s_sate_rgb_00.png' % name.replace('_00', '') for name in names]
	idx = [i for i in range(len(names))]

	random.seed(7)
	li = []
	for _ in range(opt.epoch):
		random.shuffle(idx)
		li += idx
	rounds = int(len(li) / opt.batch_size)

	for i in range(rounds):
		beg = i * opt.batch_size
		end = beg + opt.batch_size
		choose = li[beg: end]

		optimizer.zero_grad()
		z_target = np.array([vectors[choose[j]] for j in range(opt.batch_size)])
		z_target = torch.from_numpy(z_target).to(opt.device)
		sate_rgb = [transforms(Image.open(paths[choose[j]])) for j in range(opt.batch_size)]
		sate_rgb = torch.stack(sate_rgb).to(opt.device)
		z_pred, _ = netE(sate_rgb)

		loss = criterionL1(z_pred, z_target)
		loss.backward()
		print('Round %d/%d, Loss %.3lf' % (i, rounds, loss.item()))
		optimizer.step()

		checkpoint = {
			'model_state_dict': netE.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}

		if i % 5000 == 0:
			torch.save(checkpoint, './sate_encoder/sate_encoder_aug_expand_%d.pth' % int(i / 5000))
			torch.save(checkpoint, './sate_encoder/sate_encoder_aug_expand_latest.pth')




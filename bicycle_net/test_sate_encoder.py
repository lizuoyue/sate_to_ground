import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html

from train_sate_encoder import option as SateOption
from models import networks
import torch
import torchvision
import random
import numpy as np
from PIL import Image

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

######
sateOpt = SateOption()
sateE = networks.define_E(sateOpt.output_nc, sateOpt.nz, sateOpt.nef, netE=sateOpt.netE, norm=sateOpt.norm, nl=sateOpt.nl,
                            init_type=sateOpt.init_type, init_gain=sateOpt.init_gain, gpu_ids=sateOpt.gpu_ids, vaeLike=sateOpt.use_vae)
sateCheckpoint = torch.load('sate_encoder/sate_encoder_latest.pth')
sateE.load_state_dict(sateCheckpoint['model_state_dict'])
sateE.eval()
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
######

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz, seed=7)

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    key = os.path.basename(data['A_paths'][0]).split('_')[0]
    sate_path = data['A_paths'][0].replace('val', 'robustness2_changed/sate').replace('.png', '_sate_rgb.png')
    # sate_path = '/home/zoli/xiaohu_new_data/test_augment/test_0/%s_sate_rgb_00.png' % key
    print('process input image %3.3d/%3.3d, %s' % (i, opt.num_test, key))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground_truth', 'encoded']
            ###
            with torch.no_grad():
                sate_rgb = transforms(Image.open(sate_path)).to(sateOpt.device)
                z0, _ = sateE(sate_rgb.unsqueeze(0))
                torch.manual_seed(nn)
                z0 += torch.randn(opt.nz) / 4.0
                images.append(model.netG(model.real_A, z0))
                names.append('encoded_satellite')
            ###
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size_w)

webpage.save()

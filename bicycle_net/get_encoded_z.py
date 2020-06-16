import os
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np

# options
testOpt = TestOptions().parse()
trainOpt = TrainOptions().parse()
trainOpt.num_threads = 1   # test code only supports num_threads=1
trainOpt.batch_size = 1   # test code only supports batch_size=1
trainOpt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(trainOpt)
model = create_model(testOpt)
model.setup(testOpt)
model.eval()
print('Loading model %s' % testOpt.model)

d = {}
# test stage
for i, data in enumerate(dataset):
    key = os.path.basename(data['A_paths'][0]).replace('.png', '')
    if not key.endswith('00'):
        continue
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, len(dataset)))
    z, _ = model.netE(model.real_B)
    print(z.shape)
    d[key] = z.cpu().detach().numpy()[0]

np.save('encoded_z_%s.npy' % os.path.basename(trainOpt.dataroot), d)

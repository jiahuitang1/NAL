# import pyiqa
from PIL import Image
import argparse
import numpy as np
import os
from net.models import NalSuper
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
from torchvision import transforms

parser=argparse.ArgumentParser()
parser.add_argument('--save',type=bool,default=True)
opt=parser.parse_args()
gps=3
blocks=15
img_dir='/data1/tjh/NPE'
result_dir='/data1/tjh/experiment_result/NPE'
print("result_dir:",result_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
model_dir = '/home/tjh/NalSuper/net/trained_models/clip_train_ffa_3_15.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=NalSuper(gps=gps,blocks=blocks)
net=nn.DataParallel(net)
net.eval()
net.load_state_dict(ckp['model'])
net.cuda()
for im in os.listdir(img_dir):
    low_light = Image.open(img_dir+'/'+ im)
    low_light_index = img_dir + im
    id_result = low_light_index.split('/')[-1].split('.')[0]
    id = low_light_index.split('/')[-1]
    # TODO
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(low_light)[None,::]
    haze1 = haze1.cuda()

    testtransform = transforms.Compose([
        transforms.Resize([400, 600]),
    ])
    result_tensor =testtransform(haze1)

    with torch.no_grad():
        pred = net(result_tensor)
    if opt.save:
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts, result_dir +'/'+  im.split('.')[0] + '.png')
        print('save image '  + im.split('.')[0] + '.png')
# unique_score = model_unique(result_dir)

# print(f'unique:{unique_score:.4f}')
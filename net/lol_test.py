import os,argparse
import numpy as np
from PIL import Image
from mpmath.identification import transforms
from metrics import psnr,ssim,lpips
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
# import pyiqa
abs=os.getcwd()+'/'


parser=argparse.ArgumentParser()
parser.add_argument('--save',type=bool,default=True,help='whether to save images')
opt=parser.parse_args()

gps=3
blocks=15
img_dir='/home/tjh/dataset/LOL-v1/eval15/input/'
result_dir='/home/tjh/NalSuper/demo/'
gt_dir = '/home/tjh/dataset/LOL-v1/eval15/gt'
print("result_dir:",result_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# model_dir=abs+f'trained_models/{dataset}_train_ffa_{gps}_{bl ocks}.pk'
# model_dir = '/home/tjh/Deep_Learning/FFA-Net-master/net/trained_models/its_train_ffa_3_20.pk'
model_dir = '/home/tjh/NalSuper/net/trained_models/clip_train_ffa_3_15.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=NalSuper(gps=gps,blocks=blocks)
net=nn.DataParallel(net)
net.eval()
net.load_state_dict(ckp['model'])
net.cuda()
ssims = []
psnrs = []
lpips_list = []

for im in os.listdir(img_dir):
    # print(f'\r {im}',end='',flush=True)
    low_light = Image.open(img_dir+im)
    low_light_index = img_dir + im
    id_result = low_light_index.split('/')[-1].split('.')[0]
    id = low_light_index.split('/')[-1]
    clear_name = id
    clear = Image.open(os.path.join(gt_dir, clear_name))
    clear_np = np.array(clear).astype(np.float32)
    clear_np = clear_np/255.0
    gt_tensor = torch.tensor(clear_np).unsqueeze(0).permute((0, 3, 1, 2)).cuda()
    # TODO
    # # result = Image.open(os.path.join(result_dir, clear_name ))
    # result = Image.open(os.path.join(result_dir, id_result+'.png'))
    # result_np = np.array(result).astype(np.float32)
    # result_np = result_np / 255.0
    # result_tensor = torch.tensor(result_np).unsqueeze(0).permute((0, 3, 1, 2)).cuda()

    #
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(low_light)[None,::]
    # haze_no=tfs.ToTensor()(low_light)[None,::]
    haze1 = haze1.cuda()
    # TODO
    # if result_tensor.shape != gt_tensor.shape:
    #     print(result_tensor.shape)
    #     print(gt_tensor.shape)
    #     testtransform = transforms.Compose([
    #         transforms.Resize([400, 600]),
    #     ])
    #     result_tensor =testtransform(result_tensor)
    #
    with torch.no_grad():
        pred = net(haze1)
    ssim1 = ssim(pred, gt_tensor).item()
    psnr1 = psnr(pred, gt_tensor)
    lpips1 = lpips(pred, gt_tensor)
    print('img:{},psnr:{:.4f},ssim:{:.4f},lpips:{:.4f}'.format(clear_name, psnr1, ssim1,lpips1))
    ssims.append(ssim1)
    psnrs.append(psnr1)
    lpips_list.append(lpips1)

    if opt.save:
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts,result_dir+im.split('.')[0]+'.png')
print(f'ssim:{np.mean(ssims):.4f}| psnr:{np.mean(psnrs):.4f} | lpips:{np.mean(lpips_list):.4f}')
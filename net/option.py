import os, argparse

import torch, warnings

warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int, default=1000)#72750
parser.add_argument('--device',type=str, default='Automatic detection')
parser.add_argument('--resume',type=bool, default=True)
parser.add_argument('--eval_step', type=int, default=30) #30 epochs
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir', type=str, default='./trained_models/')
parser.add_argument('--dataset', type=str, default='lol_v2_real', help='lol_v1,lol_v2_real,lol_v2_syn,SID,Fivek,ve_lol_cap,ve_lol_syn,sdsd_indoor,sdsd_outdoor,smid')
# parser.add_argument('--trainset', type=str, default='lol_v1_train')
# parser.add_argument('--testset', type=str, default='lol_v1_test')
parser.add_argument('--net', type=str, default='NalSuper')
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=15, help='residual_blocks') #20
parser.add_argument('--bs', type=int, default=1, help='batch size')#160
parser.add_argument('--crop', default=True)
parser.add_argument('--crop_size', type=int,default=128, help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--perloss', action='store_true', help='perceptual loss')
parser.add_argument('--seed',type=int, default=2588)
opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
opt.trainset = opt.dataset + '_train'
opt.testset = opt.dataset + '_test'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)
# opt.model_dir='/home/tjh/Deep_Learning/NalSuper/net/trained_models/'+model_name+'.pk'
opt.model_dir='/home/tjh/NalSuper/net/trained_models/'+model_name+'.pk'

# TODO
# opt.model_dir= '/home/tjh/Deep_Learning/FFA-Net-master/net/trained_models/clip_train_ffa_3_15.pk'
log_dir='/data1/tjh/experiment_result/NalSuper/'+'logs/'+model_name
print(opt)
print('model_dir:', opt.model_dir)
if not os.path.exists(log_dir):
	os.mkdir(log_dir)

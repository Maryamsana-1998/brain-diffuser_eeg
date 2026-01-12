import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
from PIL import Image
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,3,4,5,6,7,8,9,10]

cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)

train_path = '/home/sanama/EEG_dataset/things-eeg/Preprocessed_data_250Hz_whiten/sub-{:02d}/train.pt'.format(sub)
train_data = torch.load(train_path, map_location="cpu")
train_imgs = train_data['img']

test_path = '/home/sanama/EEG_dataset/things-eeg/Preprocessed_data_250Hz_whiten/sub-{:02d}/test.pt'.format(sub)
test_data = torch.load(test_path, map_location="cpu")
test_caps = test_data['text']
test_imgs = test_data['img']

class batch_generator_external_images(Dataset):

    def __init__(self, imgs):
        self.im = imgs
        self.image_root = Path("../EEG_dataset/things-eeg/Image_set/")
    def __getitem__(self,idx):
        p = self.im[idx][0]
        # p might be a torch string, numpy string, or python string
        p = str(p)
        img_path = self.image_root / p
        img = Image.open(img_path).convert("RGB")
        img = F.resize(img, (512, 512))    # resize expects PIL
        img = F.to_tensor(img).float()     # PIL → torch.Tensor in [0,1]
        img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)

batch_size=1
train_images = batch_generator_external_images(train_imgs)
test_images = batch_generator_external_images(test_imgs)

trainloader = DataLoader(train_images,batch_size,shuffle=False)
testloader = DataLoader(test_images,batch_size,shuffle=False)

num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)

train_clip = np.zeros((num_train,num_embed,num_features))
test_clip = np.zeros((num_test,num_embed,num_features))

with torch.no_grad():
    for i,cin in enumerate(testloader):
        print(i)
        #ctemp = cin*2 - 1
        c = net.clip_encode_vision(cin)
        # print("encode shape", c[0].shape)
        test_clip[i] = c[0].cpu().numpy()
    
    np.save('data/extracted_features/eeg/subj{:02d}_clipvision_test.npy'.format(sub),test_clip)
        
    for i,cin in enumerate(trainloader):
        #ctemp = cin*2 - 1
        c = net.clip_encode_vision(cin)
        train_clip[i] = c[0].cpu().numpy()
    np.save('data/extracted_features/eeg/subj{:02d}_clipvision_train.npy'.format(sub),train_clip)


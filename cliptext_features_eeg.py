import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
from PIL import Image
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T

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
train_caps = train_data['text']

test_path = '/home/sanama/EEG_dataset/things-eeg/Preprocessed_data_250Hz_whiten/sub-{:02d}/test.pt'.format(sub)
test_data = torch.load(test_path, map_location="cpu")
test_caps = test_data['text']

print(len(test_caps),' test caps lenght')
print(len(train_caps),' train caps lenght')
num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

train_clip = np.zeros((num_train,num_embed, num_features))
test_clip = np.zeros((num_test,num_embed, num_features))
with torch.no_grad():
    for i,annots in enumerate(test_caps):
        cin = list(annots[annots!=''])
        print(i)
        c = net.clip_encode_text(cin)
        test_clip[i] = c.to('cpu').numpy().mean(0)
    
    np.save('data/extracted_features/eeg/subj{:02d}_cliptext_test.npy'.format(sub),test_clip)
        
    for i,annots in enumerate(train_caps):
        cin = list(annots[annots!=''])
        # print(i)
        c = net.clip_encode_text(cin)
        train_clip[i] = c.to('cpu').numpy().mean(0)
    np.save('data/extracted_features/eeg/subj{:02d}_cliptext_train.npy'.format(sub),train_clip)
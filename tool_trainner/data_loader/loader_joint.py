import numpy as np
from torch.utils.data import Dataset
import math
from path import Path

import sys # load img modifer
sys.path.append("..")
import random
import torch
from data_loader.utils import load_as_float, DataGenerator, ETH_img_load, syn_img_load

class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, sequence_len=2, dataset='syn', aug=False, is_train=True):
        super(DataSetLoader, self).__init__()
        if dataset == 'ETH':
            if is_train : 
                sections = ['ZurichRAW/train/']
            else:
                sections = ['ZurichRAW/test/']
        elif dataset == 'syn':
            if is_train : 
                sections = ['syn_dataset/train/']
            else:
                sections = ['syn_dataset/test/']

        self.img_rgb_list = []
        self.img_raw_list = []

        for section in sections : 
            if dataset == 'ETH':
                dir_rgb = Path(dataset_dir+'/'+section+'/canon/')
                dir_raw = Path(dataset_dir+'/'+section+'/huawei_raw/')
                self.img_rgb_list.append(sorted(dir_rgb.files('*.jpg')))
                self.img_raw_list.append(sorted(dir_raw.files('*.png')))
            elif dataset == 'syn':
                dir_rgb = Path(dataset_dir+'/'+section+'/rgb/')
                dir_raw = Path(dataset_dir+'/'+section+'/raw/')
                self.img_rgb_list.append(sorted(dir_rgb.files('*.png')))
                self.img_raw_list.append(sorted(dir_raw.files('*.png')))

        self.img_rgb_list = np.concatenate(self.img_rgb_list, axis=0)
        self.img_raw_list = np.concatenate(self.img_raw_list, axis=0)

        self.DGL  = DataGenerator(lv='low',aug=aug)
        self.DGH  = DataGenerator(lv='high',aug=aug)

        self.sequence_len = sequence_len
        self.do_act = []


        self.actions_dict = {'DeNoiseNet_L3' : self.DGL.addnoise,    'DeNoiseNet_L8' : self.DGH.addnoise, \
                             'SRNet_L3'      : self.DGL.make_LR,     'SRNet_L8'      : self.DGH.make_LR, \
                             'DeJpegNet_L3'  : self.DGL.make_jpeg,   'DeJpegNet_L8'  : self.DGH.make_jpeg, \
                             'DeBlurNet_L3'  : self.DGL.addblur_gus, 'DeBlurNet_L8'  : self.DGH.addblur_gus,\
                             'ExposureNet_L3': self.DGL.exp_gam,     'ExposureNet_L8': self.DGH.exp_gam,\
                             'CTCNet_L3'     : self.DGL.raw2rgb,     'CTCNet_L8'     : self.DGH.raw2rgb,}
        self.actions_name = ['DeNoiseNet_L3', 'DeNoiseNet_L8', 'SRNet_L3', 'SRNet_L8', 'DeJpegNet_L3', 'DeJpegNet_L8',\
                             'DeBlurNet_L3', 'DeBlurNet_L8', 'ExposureNet_L3', 'ExposureNet_L8', 'CTCNet_L3', 'CTCNet_L8']

        self.raw_flag = False
        self.is_train = is_train
        self.set_actions()
        if dataset == 'ETH':
            self.reshaper = ETH_img_load
        elif dataset == 'syn':
            self.reshaper = syn_img_load

    def __getitem__(self, idx):
        img_raw_ = load_as_float(self.img_raw_list[idx])
        init_img_ = load_as_float(self.img_rgb_list[idx])
        img_raw, init_img = self.reshaper(img_raw_, init_img_, is_train=self.is_train)
        if self.raw_flag : 
            self.DGL.raw = img_raw
            self.DGH.raw = img_raw

        H_,W_,C_ = init_img.shape
        outimgs = np.zeros((self.sequence_len, C_, H_, W_))
        img = init_img.copy()
        for i in range(self.sequence_len):
            outimg = self.actions_dict[self.do_act[i]](img)
            outimgs[i] = torch.from_numpy(outimg.transpose(2,0,1)).float() / 255.
            img = outimg
        initial_img = np.expand_dims(torch.from_numpy(init_img.transpose(2,0,1)).float() / 255., axis=0)
        outimgs = np.concatenate((initial_img, outimgs), axis=0)
        return outimgs, self.do_act

    def __len__(self):
        return len(self.img_raw_list)

    def set_actions(self) : 
        if random.random() < 0.5:
            self.raw_flag = True
            if random.random() < 0.5: 
                self.do_act = [self.actions_name[-1]] # CTCNet first
            else:
                self.do_act = [self.actions_name[-2]] # CTCNet first
            for _ in range(self.sequence_len-1):
                idx = random.randint(0,len(self.actions_name)-3) # except CTC Net
                self.do_act.append(self.actions_name[idx]) 
        else:
            self.raw_flag = False
            self.do_act = []
            for _ in range(self.sequence_len):
                idx = random.randint(0,len(self.actions_name)-3) # except CTCNet
                self.do_act.append(self.actions_name[idx]) 

import h5py
import numpy as np
from torch.utils.data import Dataset
import math
from path import Path
import random
from common.ImgModifier import ImgModifier

from data_loader.utils import load_as_float, DataGenerator, ETH_img_load, syn_img_load, WB_img_load

class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, tf=None, dataset='ETH', mode='raw', level='low', aug=False, is_train=True):
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
        else:
            if is_train : 
                sections = ['WB/train/']
            else:
                sections = ['WB/test/']

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
            else : # Set5 dataset for whitebalance traning
                img_dir_in   = Path(dataset_dir+'/'+section+'/in_image/')
                img_dir_gt   = Path(dataset_dir+'/'+section+'/gt_image/')
                self.img_raw_list.append(sorted(img_dir_in.files('*.png')))
                self.img_rgb_list.append(sorted(img_dir_gt.files('*.png')))

        self.img_rgb_list = np.concatenate(self.img_rgb_list, axis=0)
        self.img_raw_list = np.concatenate(self.img_raw_list, axis=0)

        self.ImgMod_base = ImgModifier('basic')
        self.ImgMod      = ImgModifier(level)
        self.aug = aug

        self.tf = tf
        self.is_train = is_train
        self.data_generator, self.ftype = self.set_modifer(mode)
        if dataset == 'ETH':
            self.reshaper = ETH_img_load
        elif dataset == 'syn':
            self.reshaper = syn_img_load
        else :
            self.reshaper = WB_img_load

    def __getitem__(self, idx):
        img_raw_ = load_as_float(self.img_raw_list[idx])
        img_rgb_ = load_as_float(self.img_rgb_list[idx])

        img_raw, img_rgb = self.reshaper(img_raw_, img_rgb_,is_train=self.is_train)
        img_in , img_gt  = self.data_generator(img_raw, img_rgb)

        imgs    = self.tf([img_in] + [img_gt])
        imgs_in = imgs[0]
        imgs_gt = imgs[1]
        return imgs_in.clamp(0.,1.), imgs_gt

    def __len__(self):
        return len(self.img_rgb_list)

    def set_modifer(self, mode) :
        if mode == 'raw' :
            return self.raw2rgb, None
        elif mode == 'noise' :
            return self.addnoise, 0
        elif mode == 'boxblur' :
            return self.addblur, 0
        elif mode == 'gusblur' :
            return self.addblur, 1
        elif mode == 'motionblur' :
             return self.addblur, 2
        elif mode == 'jpeg' :
            return self.make_jpeg, 255
        elif mode == 'sr' :
            return self.make_LR, 255
        elif mode == 'bright_add' :
            return self.change_bright, 0
        elif mode == 'bright_mul' :
            return self.change_bright, 1
        elif mode == 'bright_gam' :
            return self.change_bright, 2
        elif mode == 'wb' :
            return self.white_balance, 255

    def raw2rgb(self, img_raw, img_rgb):
        return img_raw, img_rgb

    def addnoise(self, img_raw, img_rgb):
        if random.random() < 0.5:
            img_gt = img_raw
        else:
            img_gt = img_rgb

        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_gt,'noise')
            img_in = self.ImgMod.addnoise(img_in) 
        else:
            img_in = self.ImgMod.addnoise(img_gt) 

        return img_in, img_gt

    def addblur(self, img_raw, img_rgb):
        if random.random() < 0.5:
            img_gt = img_raw
        else:
            img_gt = img_rgb

        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_gt,'blur')
            img_in = self.ImgMod.addblur(img_in, Ftype=self.ftype) 
        else:
            img_in = self.ImgMod.addblur(img_gt, Ftype=self.ftype)

        return img_in, img_gt

    def make_LR(self, img_raw, img_rgb):
        if random.random() < 0.5:
            img_gt = img_raw
        else:
            img_gt = img_rgb

        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_gt,'sr')
            img_in = self.ImgMod.make_lowresol(img_in) 
        else:
            img_in = self.ImgMod.make_lowresol(img_gt)

        return img_in, img_gt

    def make_jpeg(self, img_raw, img_rgb):
        if random.random() < 0.5:
            img_gt = img_raw
        else:
            img_gt = img_rgb

        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_gt,'jpeg')
            img_in = self.ImgMod.jpg_comp(img_in) 
        else:
            img_in = self.ImgMod.jpg_comp(img_gt)

        return img_in, img_gt

    def change_bright(self, img_raw, img_rgb):
        img_gt = (img_rgb + (154 - img_rgb.mean())) # we set a bit high mid-tone of exposure as a GT exposure.

        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_gt,'jpeg')
            img_in = self.ImgMod.change_brightness(img_in, Ftype=self.ftype) # add, mul, gam
        else:
            img_in = self.ImgMod.change_brightness(img_gt, Ftype=self.ftype)

        return img_in, img_gt

    def white_balance(self, img_in, img_gt):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'wb')

        return img_in, img_gt

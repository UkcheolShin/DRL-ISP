import numpy as np
from imageio import imread
import cv2
import math

import sys # load img modifer
sys.path.append("..")
from common.ImgModifier import ImgModifier
import random

## DRL-ISP psnr calculation function
# @brief 
def psnr_cal(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr

def load_as_float(path):
    img = imread(path)
    if img.dtype == 'uint16' : 
        img = img/1023.*255
    return img.astype(np.float32)

class DataGenerator:
    def __init__(self, lv, aug=False):
        super(DataGenerator, self).__init__()
        self.ImgMod_base = ImgModifier('basic')
        self.ImgMod      = ImgModifier(lv)
        self.aug         = aug
        self.raw         = None

    def raw2rgb(self, img_in):
        return self.raw

    def addnoise(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'noise')
            img_out = self.ImgMod.addnoise(img_in, 0) 
        else:
            img_out = self.ImgMod.addnoise(img_in, 0) 
        return img_out

    def addblur_box(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'blur')
            img_out = self.ImgMod.addblur(img_in, 0) 
        else:
            img_out = self.ImgMod.addblur(img_in, 0)
        return img_out

    def addblur_gus(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'blur')
            img_out = self.ImgMod.addblur(img_in, 1) 
        else:
            img_out = self.ImgMod.addblur(img_in, 1)
        return img_out

    def addblur_mtn(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'blur')
            img_out = self.ImgMod.addblur(img_in, 2) 
        else:
            img_out = self.ImgMod.addblur(img_in, 2)
        return img_out

    def make_LR(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'sr')
            img_out = self.ImgMod.make_lowresol(img_in) 
        else:
            img_out = self.ImgMod.make_lowresol(img_in)
        return img_out

    def make_jpeg(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'jpeg')
            img_out = self.ImgMod.jpg_comp(img_in) 
        else:
            img_out = self.ImgMod.jpg_comp(img_in)
        return img_out

    def exp_add(self, img_in):
        # img_gt = (img_rgb + (154 - img_rgb.mean()))
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'jpeg')
            img_out = self.ImgMod.change_brightness(img_in, Ftype=0) # add, mul, gam
        else:
            img_out = self.ImgMod.change_brightness(img_in, Ftype=0)
        return img_out

    def exp_mul(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'jpeg')
            img_out = self.ImgMod.change_brightness(img_in, Ftype=1) # add, mul, gam
        else:
            img_out = self.ImgMod.change_brightness(img_in, Ftype=1)
        return img_out

    def exp_gam(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'jpeg')
            img_out = self.ImgMod.change_brightness(img_in, Ftype=2) # add, mul, gam
        else:
            img_out = self.ImgMod.change_brightness(img_in, Ftype=2)
        return img_out

def ETH_img_load(img_raw, img_rgb, is_train=True):    
    H_,W_,_ = img_rgb.shape
    img_rgb_ = np.zeros((H_,W_,4))
    img_rgb_[:,:,0] = img_rgb[:,:,0]
    img_rgb_[:,:,1] = img_rgb[:,:,1]
    img_rgb_[:,:,2] = img_rgb[:,:,1].copy()
    img_rgb_[:,:,3] = img_rgb[:,:,2]

    img_raw_ = np.zeros((H_//2,W_//2,4)) # HxWx3 ==> H/2xW/2x4
    img_raw_[:,:,0] = img_raw[1::2, 1::2] # R
    img_raw_[:,:,1] = img_raw[1::2, 0::2] # G1
    img_raw_[:,:,2] = img_raw[0::2, 1::2] # G2 
    img_raw_[:,:,3] = img_raw[0::2, 0::2] # B      

    # resize images to reduce training time
    # ETH 448x448 --> 112x112
    if is_train : 
        img_rgb   = cv2.resize(img_rgb_, (W_//4, H_//4), interpolation=cv2.INTER_CUBIC)
        img_raw   = cv2.resize(img_raw_, (W_//4, H_//4), interpolation=cv2.INTER_CUBIC)
    else:
        img_rgb   = img_rgb_
        img_raw   = cv2.resize(img_raw_, (W_, H_), interpolation=cv2.INTER_CUBIC)

    return img_raw, img_rgb

def syn_img_load(img_raw, img_rgb, is_train=True):    
    H_,W_,_ = img_rgb.shape
    img_rgb_ = np.zeros((H_,W_,4))
    img_rgb_[:,:,0] = img_rgb[:,:,0]
    img_rgb_[:,:,1] = img_rgb[:,:,1]
    img_rgb_[:,:,2] = img_rgb[:,:,1].copy()
    img_rgb_[:,:,3] = img_rgb[:,:,2]

    img_raw_ = np.zeros((H_//2,W_//2,4)) # HxWx3 ==> H/2xW/2x4
    img_raw_[:,:,0] = img_raw[1::2, 0::2] # R
    img_raw_[:,:,1] = img_raw[0::2, 0::2] # G1
    img_raw_[:,:,2] = img_raw[1::2, 1::2] # G2 
    img_raw_[:,:,3] = img_raw[0::2, 1::2] # B

    img_raw_   = cv2.resize(img_raw_, (W_, H_), interpolation=cv2.INTER_CUBIC)

    img_rgb_ = img_rgb_[:H_//16*16, :W_//16*16, :]
    img_raw_ = img_raw_[:H_//16*16, :W_//16*16, :]

    if is_train : 
        # resize images to reduce training time
        # DIV2K 336x508, KITTI 368x1232, COCO 416x640 ==> 336x336, 368x368, 416x416 
        # crop and resize to 112x112
        H_,W_,_ = img_rgb_.shape
        lower = np.min((H_,W_))
        offset_y = np.random.randint(H_ - lower + 1)
        offset_x = np.random.randint(W_ - lower + 1)
        cropped_img_rgb_ = img_rgb_[offset_y:offset_y + lower, offset_x:offset_x + lower, :] 
        cropped_img_raw_ = img_raw_[offset_y:offset_y + lower, offset_x:offset_x + lower, :] 

        img_rgb   = cv2.resize(cropped_img_rgb_, (112, 112), interpolation=cv2.INTER_CUBIC)
        img_raw   = cv2.resize(cropped_img_raw_, (112, 112), interpolation=cv2.INTER_CUBIC)
    else:
        img_rgb   = img_rgb_
        img_raw   = img_raw_
        
    return img_raw, img_rgb


def WB_img_load(img_in, img_gt, is_train=True):    
    H_,W_,_ = img_in.shape
    img_in_ = np.zeros((H_,W_,4))
    img_in_[:,:,0] = img_in[:,:,0]
    img_in_[:,:,1] = img_in[:,:,1]
    img_in_[:,:,2] = img_in[:,:,1].copy()
    img_in_[:,:,3] = img_in[:,:,2]

    img_gt_ = np.zeros((H_,W_,4))
    img_gt_[:,:,0] = img_gt[:,:,0]
    img_gt_[:,:,1] = img_gt[:,:,1]
    img_gt_[:,:,2] = img_gt[:,:,1].copy()
    img_gt_[:,:,3] = img_gt[:,:,2]

    if H_//16*16 != H_ :
        img_in = img_in[:H_//16*16, :, :]
        img_gt = img_gt[:H_//16*16, :, :]
    if W_//16*16 != W_ :
        img_in = img_in[:, :W_//16*16, :]
        img_gt = img_gt[:, :W_//16*16, :]

    if is_train : 
        # resize images to reduce training time
        H_,W_,_ = img_in.shape
        lower = np.min((H_,W_))
        offset_y = np.random.randint(H_ - lower + 1)
        offset_x = np.random.randint(W_ - lower + 1)
        cropped_img_in = img_in_[offset_y:offset_y + lower, offset_x:offset_x + lower, :] 
        cropped_img_gt = img_gt_[offset_y:offset_y + lower, offset_x:offset_x + lower, :] 

        img_in   = cv2.resize(cropped_img_in, (112, 112), interpolation=cv2.INTER_CUBIC)
        img_gt   = cv2.resize(cropped_img_gt, (112, 112), interpolation=cv2.INTER_CUBIC)
    else:
        img_rgb   = img_rgb_
        img_raw   = img_raw_

    return img_in, img_gt

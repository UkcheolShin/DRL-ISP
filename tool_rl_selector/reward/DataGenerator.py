import cv2

import sys # load img modifer
sys.path.append("..")
from common.ImgModifier import ImgModifier
import random

class DataGenerator:
    def __init__(self, lv, aug=False):
        super(DataGenerator, self).__init__()
        self.ImgMod_base = ImgModifier('basic')
        self.ImgMod      = ImgModifier(lv)
        self.aug         = aug
        self.raw         = None

    def raw_aug(self, img_in) :
        img_out = self.ImgMod.addblur(img_in) 
        img_out = self.ImgMod.addnoise(img_out) 
        return img_out

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

    def make_LR_x2(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'sr')
            img_out = self.ImgMod.make_lowresol(img_in, scale=2) 
        else:
            img_out = self.ImgMod.make_lowresol(img_in, scale=2)
        return img_out

    def make_LR_x3(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'sr')
            img_out = self.ImgMod.make_lowresol(img_in, scale=3) 
        else:
            img_out = self.ImgMod.make_lowresol(img_in, scale=3)
        return img_out

    def make_LR_x4(self, img_in):
        if self.aug :
            img_in = self.ImgMod_base.add_basic_effect(img_in,'sr')
            img_out = self.ImgMod.make_lowresol(img_in, scale=4) 
        else:
            img_out = self.ImgMod.make_lowresol(img_in, scale=4)
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

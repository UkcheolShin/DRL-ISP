import torch
from torch import nn
import numpy as np
import random
import math

import cv2
import kornia
import kornia.color as cc
import kornia.enhance as eh
import kornia.filters as ft

class TraToolsConfig(object):
    def __init__(self, continuous=False, min_value=0.0, max_value=1.0, factors=(0, )):
        self.is_continuous = continuous
        self.min = float(min_value)
        self.max = float(max_value)
        self.default_factors = factors

def bayer_to_rgb(inputs):
    B_, C_,H_,W_ = inputs.shape
    inputs_ = torch.cuda.FloatTensor(B_, 3, H_,W_, device=inputs.device)
    inputs_[:, 0,:,:] = inputs[:, 0,:,:]
    inputs_[:, 1,:,:] = (inputs[:, 1,:,:] + inputs[:, 2,:,:])/2
    inputs_[:, 2,:,:] = inputs[:, 3,:,:]
    return inputs_

def rgb_to_bayer(inputs):
    B_, C_,H_,W_ = inputs.shape
    inputs_ = torch.cuda.FloatTensor(B_, 4, H_,W_, device=inputs.device)
    inputs_[:, 0,:,:] = inputs[:, 0,:,:]
    inputs_[:, 1,:,:] = inputs[:, 1,:,:]
    inputs_[:, 2,:,:] = inputs[:, 1,:,:].clone()
    inputs_[:, 3,:,:] = inputs[:, 2,:,:]
    return inputs_

class TRA_TOOLS(object):    
    def __init__(self):
        super(TRA_TOOLS, self).__init__()
        # color space conversion
        # https://kornia.readthedocs.io/en/latest/color.html
        self.config = {
            'add_brightness': TraToolsConfig(True, 0, 1, factors=(0.05, 0.15, 0.3)),
            'sub_brightness': TraToolsConfig(True, 0, 1, factors=(0.05, 0.15, 0.3)),
            'add_contrast': TraToolsConfig(True, 0, 1, factors=(1.25, 1.5, 2.0)),
            'sub_contrast': TraToolsConfig(True, 0, 1, factors=(1.25, 1.5, 2.0)),
            'add_gamma': TraToolsConfig(True, 0, 1, factors=(0.7, 0.45, 0.30)),
            'sub_gamma': TraToolsConfig(True, 0, 1, factors=(1.2, 1.45, 1.80)),
            'add_hue': TraToolsConfig(True, 0, math.pi, factors=(0.2, 0.5, 1.0)),
            'sub_hue': TraToolsConfig(True, 0, math.pi, factors=(0.2, 0.5, 1.0)),
            'add_saturation': TraToolsConfig(True, 0, 1, factors=(1.2, 1.4, 1.8)),
            'sub_saturation': TraToolsConfig(True, 0, 1, factors=(0.8, 0.6, 0.4)),
            # 'sharpenning': TraToolsConfig(False, 0, 1, factors=(1.0)),
            # 'blur_box': TraToolsConfig(False, 0, 1, factors=(3)),
            # 'blur_gaussian': TraToolsConfig(False, 0, 1, factors=(3)),
            # 'blur_bilateral': TraToolsConfig(False, 0, 1, factors=(3)),
            # 'hist_equalize': TraToolsConfig(False, 0, 1, factors=(None, )),
            # 'hist_adap_equalize': TraToolsConfig(False, 0, 1, factors=(None, )),
            # 'white_balance': TraToolsConfig(False, 0, 1, factors=(None, )),
        }
        for x,y in TRA_TOOLS.__dict__.items() :
            if (type(y) == type(lambda:0)) and not x.startswith('_'):# from types import FunctionType, lambda:0 ->FunctionType :
                if x not in self.config:
                    self.config[x] = TraToolsConfig()

    def add_brightness(self, img_in, factor=0.15) :
        # adjust brightness 
        # torch range 0~1,
        # img_in + bright_factor (each channel)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        bright_factor = +factor # similar +10 in 255 range
        img_out = eh.adjust_brightness(img_in, bright_factor)
        
        return img_out.clamp(0.0, 1.0)

    def sub_brightness(self, img_in, factor=0.15) :
        # adjust brightness 
        # torch range 0~1,
        # img_in + bright_factor (each channel)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        bright_factor = -factor # similar +10 in 255 range
        img_out = eh.adjust_brightness(img_in, bright_factor)
        
        return img_out.clamp(0.0, 1.0)

    def add_contrast(self, img_in, factor=1.25) :
        # adjust contrast 
        # img_in * contrast_factor (each channel)
        # < 1 : darker, 1 : doesn't change at al. > 1 : brighter
        # the range should be 1 ~ 
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        contrast_factor = factor
        img_out = eh.adjust_contrast(img_in, contrast_factor)
        
        return img_out.clamp(0.0, 1.0)

    def sub_contrast(self, img_in, factor=1.25) :
        # adjust contrast 
        # img_in * contrast_factor (each channel)
        # < 1 : darker, 1 : doesn't change at al. > 1 : brighter
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        contrast_factor = 1/factor
        img_out = eh.adjust_contrast(img_in, contrast_factor)
        
        return img_out.clamp(0.0, 1.0)

    def add_gamma(self, img_in, factor=0.45) :
        # adjust gamma 
        # gain*img_in ^ gamma_factor (each channel)
        # < 1 : brighter, 1 : doesn't change at al. > 1 : darker
        # the range should be 1 ~ 
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        gamma = factor
        gain = 1.2
        img_out = eh.adjust_gamma(img_in, gamma, gain)
        
        return img_out.clamp(0.0, 1.0)

    def sub_gamma(self, img_in, factor=1.45) :
        # adjust gamma 
        # torch range 0~1,
        # gain*img_in ^ gamma_factor (each channel)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        gamma = factor
        gain = 1.2
        img_out = eh.adjust_gamma(img_in, gamma, gain)
    
        return img_out.clamp(0.0, 1.0)

    def add_hue(self, img_in, factor=0.20) :
        # adjust hue 
        # torch range -3.14~3.14,
        # rgb->hsv, h = (h + hue factor)/(2*3.14...)

        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        img_in = bayer_to_rgb(img_in)
        img_out = eh.adjust_hue(img_in, factor)
        img_out = rgb_to_bayer(img_out)
        
        return img_out.clamp(0.0, 1.0)

    def sub_hue(self, img_in, factor=0.20) :
        # adjust hue 
        # torch range -3.14~3.14,
        # rgb->hsv, h = (h + hue factor)/(2*3.14...)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        hue_factor = -factor # range -3.14 ~ 3.14

        img_in = bayer_to_rgb(img_in)
        img_out = eh.adjust_hue(img_in, hue_factor)
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)

    def add_saturation(self, img_in, factor=1.4) :
        # adjust saturation 
        # torch range 0~
        # 0 : black&white image , 1: same, > 1, multiply with factor
        # rgb->hsv, s = (s * saturation factor)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        satur_factor = +factor # 0~1, 0: black, 1: original 2:saturated
        img_in = bayer_to_rgb(img_in)
        img_out = eh.adjust_saturation(img_in, satur_factor)
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)

    def sub_saturation(self, img_in, factor=0.6) :
        # adjust saturation 
        # 0 : black&white image , 1: same, > 1, multiply with factor
        # rgb->hsv, s = (s * saturation factor)
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        satur_factor = factor # 0~1, 0: black, 1: original 2:saturated
        img_in = bayer_to_rgb(img_in)
        img_out = eh.adjust_saturation(img_in, satur_factor)
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)

    def sharpenning(self, img_in, factor=1.0) :
        # sharpen image 
        # torch range 0~1,
        # shapenning kernal [1,1,1; 1,5,1; 1,1,1] / 13
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        # sharp_factor = factor # should be > 0, 0: original, 1: sharpened image, 0~1 btw : weighted sum
        # img_out = eh.sharpness(img_in.cpu(), sharp_factor)
        img_in = bayer_to_rgb(img_in)
        if len(img_in.shape) == 4:
            img_in = img_in.squeeze(0)
        img_np = (255*img_in.clamp(0.0, 1.0).cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)
        sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 
        img_out = cv2.filter2D(img_np, -1, sharpening_1)
        img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.        
        img_out = rgb_to_bayer(img_out)
        return img_out.clamp(0.0, 1.0)

    def blur_box(self, img_in, factor=3) :
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        if len(img_in.shape) == 3: # should be BCHW shape
            img_in = img_in.unsqueeze(0)
        box_blur = ft.BoxBlur((factor,factor))
        img_out = box_blur(img_in)
        
        return img_out.clamp(0.0, 1.0)

    def blur_gaussian(self, img_in, factor=3, sigma=1.5) :
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
        if len(img_in.shape) == 3:
            img_in = img_in.unsqueeze(0)
        img_out = ft.gaussian_blur2d(img_in, (factor, factor), (sigma, sigma))
    
        return img_out.clamp(0.0, 1.0)

    @torch.no_grad()
    def blur_bilateral(self, img_in, factor=3) :
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")

        img_in = bayer_to_rgb(img_in)
        if len(img_in.shape) == 4:
            img_in = img_in.squeeze(0)
        img_np = (255*img_in.clamp(0.0, 1.0).cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)
        img_out = cv2.bilateralFilter(img_np, d=15, sigmaColor=75, sigmaSpace=75)
        # outimg = cv2.bilateralFilter(img_np)
        img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.        
        img_out = rgb_to_bayer(img_out)
        return img_out.clamp(0.0, 1.0)

    # #https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
    # #https://blog.naver.com/laonple/220834097950
    # # http://www.gisdeveloper.co.kr/?p=7168
    # @torch.no_grad()
    # def denoise(self, img_in):
    #     if not torch.is_tensor(img_in):
    #         raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")
    #     if len(img_in.shape) == 4:
    #         img_in = img_in.squeeze(0)

    #     img_np = (255*img_in.cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)
    #     img_out = cv2.fastNlMeansDenoisingColored(img_np, 15,15,5,10)
    #     # img_out = cv2.fastNlMeansDenoisingColored(img_np)
    #     img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.        
    #     return img_out.clamp(0.0, 1.0)

    @torch.no_grad()
    def hist_equalize(self, img_in):
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")

        img_in = bayer_to_rgb(img_in)
        if len(img_in.shape) == 4:
            img_in = img_in.squeeze(0)
        img_np = (255*img_in.clamp(0.0, 1.0).cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)

        img_yuv = cv2.cvtColor(img_np,cv2.COLOR_RGB2YUV) # Y : intensity, u,v : color
        img_y = cv2.equalizeHist(img_yuv[:,:,0])
        img_yuv[:,:,0] = img_y
        img_out = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2RGB) # Y : intensity, u,v : color
        img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.      
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)
    
    @torch.no_grad()
    def hist_adap_equalize(self, img_in, param1 = 2.0, param2 = 8):
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")

        img_in = bayer_to_rgb(img_in)
        if len(img_in.shape) == 4:
            img_in = img_in.squeeze(0)
        img_np = (255*img_in.clamp(0.0, 1.0).cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)

        img_yuv = cv2.cvtColor(img_np,cv2.COLOR_RGB2YUV) # Y : intensity, u,v : color
        clahe = cv2.createCLAHE(clipLimit=param1, tileGridSize=(param2,param2))
        img_y = clahe.apply(img_yuv[:,:,0])

        img_yuv[:,:,0] = img_y
        img_out = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2RGB) # Y : intensity, u,v : color
        img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)

    #https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574
    # grayworld assuption, basic white-balcne method.(von kries algorithm)
    def white_balance(self, img_in):
        if not torch.is_tensor(img_in):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img_in)}")

        img_in = bayer_to_rgb(img_in)
        if len(img_in.shape) == 4:
            img_in = img_in.squeeze(0)
        img_np = (255*img_in.clamp(0.0, 1.0).cpu().detach().numpy()).astype(np.uint8).transpose(1,2,0)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        avg_a = np.average(img_lab[:, :, 1])
        avg_b = np.average(img_lab[:, :, 2])
        img_lab[:, :, 1] = img_lab[:, :, 1] - ((avg_a - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
        img_lab[:, :, 2] = img_lab[:, :, 2] - ((avg_b - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
        img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        img_out = torch.FloatTensor(img_out.transpose(2,0,1)).unsqueeze(0).to(img_in.device)/255.      
        img_out = rgb_to_bayer(img_out)

        return img_out.clamp(0.0, 1.0)
import torch
import numpy as np
import cv2
import os
import h5py
import math
import random
from imageio import imread # rgb read
from torch import nn
from math import exp
import torch.nn.functional as F
from path import Path

from reward.DataGenerator import DataGenerator
from reward.utils import ETH_img_load, syn_img_load, load_as_float
from reward import custom_transforms

## DRL-ISP psnr calculation function
# @brief 
def psnr_cal(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr

## DRL-ISP Dataloader
# @brief RL Dataloader 
# @param config
# @return None
class Data_ImgRestoration(object):
    def __init__(self, config):
        self.is_train = ~config.is_test

        # 1. Read training data
        if config.dataset == 'ETH':
            dir_rgb = Path(config.train_dir+'/ZurichRAW/train/canon/')
            dir_raw = Path(config.train_dir+'/ZurichRAW/train/huawei_raw/')
            self.train_rgb_list = sorted(dir_rgb.files('*.jpg'))
            self.train_raw_list = sorted(dir_raw.files('*.png'))
        elif config.dataset == 'syn':
            dir_rgb = Path(config.train_dir+'/syn_dataset/train/rgb/')
            dir_raw = Path(config.train_dir+'/syn_dataset/train/raw/')
            self.train_rgb_list = sorted(dir_rgb.files('*.png'))
            self.train_raw_list = sorted(dir_raw.files('*.png'))

        if config.train_dataset_len != 0:
            train_len = config.train_dataset_len
            self.train_rgb_list = self.train_rgb_list[:train_len]
            self.train_raw_list = self.train_raw_list[:train_len]

        # 2. set data index
        self.train_data_idx = 0
        self.train_data_len = len(self.train_rgb_list)
        self.train_height, self.train_width = 112,112
        self.tf = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                             custom_transforms.RandomScaleCrop(),
                                             custom_transforms.ArrayToTensor(),
                                             ])
                                                
        if config.dataset == 'ETH':
            self.reshaper = ETH_img_load
        elif config.dataset == 'syn':
            self.reshaper = syn_img_load

        self.data_generator = DataGenerator(lv='low',aug=False)

        # 2. validation data
        if config.dataset == 'ETH':
            dir_rgb = Path(config.test_dir+'/ZurichRAW/test/canon/')
            dir_raw = Path(config.test_dir+'/ZurichRAW/test/huawei_raw/')
            self.test_rgb_list = sorted(dir_rgb.files('*.jpg'))
            self.test_raw_list = sorted(dir_raw.files('*.png'))
        elif config.dataset == 'syn':
            dir_rgb = Path(config.test_dir+'/syn_dataset/test/rgb/')
            dir_raw = Path(config.test_dir+'/syn_dataset/test/raw/')
            self.test_rgb_list = sorted(dir_rgb.files('*.png'))
            self.test_raw_list = sorted(dir_raw.files('*.png'))

        # fast experiment
        if config.test_dataset_len > 0:
            test_len = config.test_dataset_len
            self.test_rgb_list = self.test_rgb_list[:test_len]
            self.test_raw_list = self.test_raw_list[:test_len]

        # current data index
        self.test_data_idx = 0
        self.test_data_len = len(self.test_rgb_list)

        _, self.test_height, self.test_width = cv2.imread(self.test_rgb_list[0]).transpose(2,0,1).shape

    def get_train_data(self) : 
        # if data index exceed the dataset range, load new dataset
        if self.train_data_idx >= self.train_data_len :
            self.train_data_idx = 0

        img_raw_ = load_as_float(self.train_raw_list[self.train_data_idx])
        img_rgb_ = load_as_float(self.train_rgb_list[self.train_data_idx])

        img_raw, img_rgb = self.reshaper(img_raw_, img_rgb_,is_train=True)
        img_raw_mod      = self.data_generator.raw_aug(img_raw)

        imgs    = self.tf([img_raw_mod] + [img_rgb])
        test_in = imgs[0]
        test_gt = imgs[1]

        self.train_data_idx += 1
        return test_in.unsqueeze(0), test_gt.unsqueeze(0)

    def get_test_data(self):
        # read images from testset folder
        if self.test_data_idx >= self.test_data_len:
            self.test_data_idx = 0
            test_done = True
        else:
            test_done = False

        img_raw_ = load_as_float(self.test_raw_list[self.test_data_idx])
        img_rgb_ = load_as_float(self.test_rgb_list[self.test_data_idx])

        img_raw, img_rgb = self.reshaper(img_raw_, img_rgb_,is_train=False)

        test_in = (torch.from_numpy(img_raw.transpose(2,0,1)).float()/255.).clamp(0.,1.)
        test_gt = (torch.from_numpy(img_rgb.transpose(2,0,1)).float()/255.).clamp(0.,1.)

        self.test_data_idx += 1
        return test_in.unsqueeze(0), test_gt.unsqueeze(0), test_done

    def get_data_shape(self):
        return None, self.train_height, self.train_width


############################################################
## DRL-ISP Rewarder
# @brief RL Reward Generator 
# @param config
# @return None
class Rewarder_ImgRestoration(object):
    def __init__(self, config):

        # enviromental parameter for training
        self.done_function = self._done_function
        self.psnr_init = 0.
        self.psnr_prev = 0.
        
        # enviromental parameter for testing
        self.done_function_test = self._done_function_with_idx
        self.psnr_init_test = 0.
        self.psnr_prev_test = 0.

        # shared enviromental parameter for training / testing
        # replace to argment.
        # reward functions
        if config.metric == 'psnr' : 
            self.metric_function = self.psnr_cal
        elif config.metric == 'ssim' : 
            self.metric_function = self.ssim_cal
        elif config.metric == 'msssim' : 
            self.metric_function = self.msssim_cal


        self.reward_function = self._step_psnr_reward
        self.get_playscore   = self.get_step_test_psnr
        self.scale = config.reward_scaler

    ## DRL-ISP train param initialize
    # @brief 
    def reset_train(self, train_data, train_data_gt) : 
        # initialize environmental param
        psnr_init = self.metric_function(train_data, train_data_gt)
        self.psnr_init      = psnr_init
        self.psnr_prev      = psnr_init
        return None

    ## DRL-ISP test param initialize
    # @brief 
    def reset_test(self, test_data, test_data_gt, return_img=False) : 
        # initialize environmental param
        self.psnr_init_test = torch.zeros(len(test_data))
        self.psnr_prev_test = torch.zeros(len(test_data))
        imgs = []
        err_imgs = []
        for k in range(len(test_data)):
            if return_img:
                init_psnr  = self.metric_function(test_data[[k], ...], test_data_gt[[k], ...])
                imgs.append(test_data)
                err_imgs.append(test_data_gt-test_data)
            else:
                init_psnr = self.metric_function(test_data[[k], ...], test_data_gt[[k], ...])

            self.psnr_init_test[k] = init_psnr
            self.psnr_prev_test[k] = init_psnr
        if return_img:
            return torch.cat(imgs, dim=0), torch.cat(err_imgs, dim=0)

        return None

    ## DRL-ISP Img restoration reward getter (train)
    # @brief 
    @torch.no_grad()
    def get_reward_train(self, data_in, data_gt):
        psnr_cur = self.metric_function(data_in, data_gt)
        reward = self.reward_function(psnr_cur)
        done = self.done_function(psnr_cur)
        self.psnr_prev = psnr_cur
        return reward, done

    ## DRL-ISP Img restoration reward getter (test)
    # @brief 
    @torch.no_grad()
    def get_reward_test(self, data_in, data_gt, idx, return_img=False):
        if return_img:
            psnr_cur = self.metric_function(data_in, data_gt)
            output_img = data_in
            err_img = data_gt - data_in
        else:
            psnr_cur = self.metric_function(data_in, data_gt)

        reward = self.reward_function(psnr_cur, idx)
        done = self.done_function_test(psnr_cur, idx)
        self.psnr_prev_test[idx] = psnr_cur
        if return_img:
            return reward, done, output_img, err_img
        return reward, done

    ## DRL-ISP Img restoration psnr getter (test)
    # @brief 
    def get_step_test_psnr(self): 
        return self.psnr_prev_test

    ## DRL-ISP inplace reward function
    # @brief decide done flag according to psnr difference, episode_step
    def _step_psnr_reward(self, psnr_cur, idx=None):
        if idx == None :
            reward = psnr_cur - self.psnr_prev
        else:
            reward = psnr_cur - self.psnr_prev_test[idx]
        return reward * self.scale

    ## DRL-ISP inplace done function
    # @brief decide done flag according to psnr difference, episode_step
    def _done_function(self, psnr_cur):
        # if psnr_cur - self.psnr_init < -5 :  # psnr is too bad
        #     return True
        return False

    ## DRL-ISP inplace done function
    # @brief decide done flag according to psnr difference, episode_step
    def _done_function_with_idx(self, psnr_cur, idx):
        # if psnr_cur - self.psnr_init_test[idx] < -5:  # psnr is too bad
        #     return True
        return False

    ## DRL-ISP psnr calculation function
    # @brief 
    def psnr_cal(self, im_input, im_label):
        loss = (im_input - im_label) ** 2
        eps = 1e-10
        loss_value = loss.mean() + eps
        psnr = 10 * math.log10(1.0 / loss_value)
        return psnr

    ## DRL-ISP psnr calculation function
    # @brief 
    def ssim_cal(self, im_input, im_label):
        ssim = SSIM_Metric(im_input, im_label)
        return ssim.mean()

    ## DRL-ISP psnr calculation function
    # @brief 
    def msssim_cal(self, im_input, im_label):
        ssim = MSSSIM_Metric(im_input, im_label)
        return ssim.mean()

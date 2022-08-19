from __future__ import division
import random
import numpy as np
import cv2
import torch
from imageio import imread

def load_as_float(path):
    img = imread(path)
    if img.dtype == 'uint16' : 
        img = img/1023.*255
    return img.astype(np.float32)
    
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


def bayer_to_rgb(inputs):
    B_, C_,H_,W_ = inputs.shape
    inputs_ = torch.cuda.FloatTensor(B_, 3, H_,W_, device=inputs.device)
    inputs_[:, 0,:,:] = inputs[:, 0,:,:]
    inputs_[:, 1,:,:] = (inputs[:, 1,:,:] + inputs[:, 2,:,:])/2
    inputs_[:, 2,:,:] = inputs[:, 3,:,:]
    return inputs_


def raw4ch(img_raw):
    H_, W_, _ = img_raw.shape

    img_raw = np.sum(img_raw, axis=2)
    R = img_raw[1::2, 0::2]
    G1 = img_raw[0::2, 0::2]
    G2 = img_raw[1::2, 1::2]
    B = img_raw[0::2, 1::2]
    H_new = G2.shape[0]
    W_new = G2.shape[1]
    img_raw_ = np.zeros((H_new, W_new, 4)) # HxWx3 ==> H/2xW/2x4
    img_raw_[:,:,0] = R[:H_new, :W_new]
    img_raw_[:,:,1] = G1[:H_new, :W_new]
    img_raw_[:,:,2] = G2[:H_new, :W_new]
    img_raw_[:,:,3] = B[:H_new, :W_new]
    img_raw_   = np.clip(cv2.resize(img_raw_, (W_, H_), interpolation=cv2.INTER_CUBIC), 0, 255)
    return img_raw_


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

def step_general_reward(img, last_img=None):
    img = img.detach().cpu()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_value = gray.mean()
    grad_img = cv2.Sobel(img, -1, 1, 1)
    img_mask = (grad_img < 25)
    masked_img = grad_img*img_mask
    noise = masked_img.mean()
    gray_coef = 1.0 #kwargs.get('gray_coef', 1.0)
    noise_coef = 0.5 #kwargs.get('noise_coef', 0.5)
    return gray_coef * (1 - 2 * float(abs(mean_value - 127.0) / 127.0)) - noise_coef * noise

from __future__ import division
import torch
import random
import numpy as np
import cv2
import kornia
import kornia.filters as ft

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
        else:
            output_images = images
        return output_images

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images):

        in_h, in_w, ch = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        if ch == 1:
            scaled_images = [np.expand_dims(cv2.resize(im, (scaled_w, scaled_h)), axis=2) for im in images]
        else :
            scaled_images = [cv2.resize(im, (scaled_w, scaled_h)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        return cropped_images

class TensorRandomOffset(object):
    """Randomly change image offsets up to -0.25~0.25 range with a probability of 0.5."""
    def __init__(self, prob=0.5, factor=0.25, scale=1.):
        self.prob = prob
        self.factor = factor
        self.scale = scale//2

    def __call__(self, images):
        if random.random() < self.prob:
            output_images = []
            offset = np.random.uniform(-self.factor,self.factor)*self.scale              
            for im in images: #CHW
                im_out = im+offset
                output_images.append(im_out)
        else:          
            output_images = images

        return output_images

class TensorRandomContrast(object):
    """Randomly add temparautre jitter to the image offsets up to -0.5~0.5 range with a probability of 0.5."""
    def __init__(self, prob=0.5, factor=0.5):
        self.prob = prob
        self.factor = factor

    def __call__(self, images):
        if random.random() < self.prob:
            output_images = []
            contrast = np.random.uniform(np.max((0,1-self.factor)), 1+self.factor)               
            for im in images: #CHW
                im_out = im*contrast
                output_images.append(im_out)
        else:          
            output_images = images

        return output_images

class TensorRandomGaussianBlur(object):
    """Randomly add temparautre jitter to the image offsets up to -0.5~0.5 range with a probability of 0.5."""
    def __init__(self, prob=0.5, sigma=0.1):
        self.prob = prob
        self.sigma = sigma
        self.ksize = 2*int(4*self.sigma + 0.5) + 1

    def __call__(self, images):
        if random.random() < self.prob:
            output_images = []
            for im in images: #CHW
                im_out = ft.gaussian_blur2d(im.unsqueeze(0), (self.ksize, self.ksize), (self.sigma, self.sigma)).squeeze(0)
                output_images.append(im_out)
        else:          
            output_images = images

        return output_images

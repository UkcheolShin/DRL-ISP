import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import kornia.color as cc
import torch.utils.model_zoo as model_zoo
import math
from .utils import conv2d_size_out

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, input_h=112, input_w=112):
        super(AlexNet, self).__init__()
        self.h = input_h
        self.w = input_w
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        feature_list = list(self.features)

        size1 = input_w
        for feat in feature_list:
            stride = getattr(feat, 'stride', None)
            kernel_size = getattr(feat, 'kernel_size', None)
            padding = getattr(feat, 'padding', None)
            if stride is None or kernel_size is None or padding is None:
                continue
            stride = stride[0] if isinstance(stride, tuple) else stride
            kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            padding = padding[0] if isinstance(padding, tuple) else padding

            size1 = conv2d_size_out(size1, kernel_size, stride, padding)
        self.feature_w = size1
        size1 = input_h
        for feat in feature_list:
            stride = getattr(feat, 'stride', None)
            kernel_size = getattr(feat, 'kernel_size', None)
            padding = getattr(feat, 'padding', None)
            if stride is None or kernel_size is None or padding is None:
                continue
            stride = stride[0] if isinstance(stride, tuple) else stride
            kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            padding = padding[0] if isinstance(padding, tuple) else padding

            size1 = conv2d_size_out(size1, kernel_size, stride, padding)
        self.feature_h = size1

    def get_output_size(self):
        return 256 * self.feature_w * self.feature_h

    def forward(self, x):
        x = F.interpolate(x, (self.h, self.w), mode='area')
        f = self.features(x)
        x = f.reshape(f.size(0), -1)
        return f, x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        pre_model = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()

        new_dict = dict()
        for k in pre_model.keys():
            if k in model_dict.keys():
                new_dict[k] = pre_model[k]
        model.load_state_dict(new_dict)
        
        for param in model.parameters():
            param.required_grad = False
        
    return model

# Histogram based feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, num_scales = 2, downsacle=2, bins_num=32):
        super(FeatureExtractor, self).__init__()
        self.num_scales = 2
        self.downscale = 2
        self.bins_num = bins_num 

    def get_output_size(self):
        return self.bins_num * 3 * 4 * 4

    def forward(self, imgs_in) :  
        # assume single-channel input
        b, _, h, w = imgs_in.size()
        # reshape odd num shape --> even num shape
        tgt_imgs = F.interpolate(imgs_in, (h//2*2, w//2*2), mode='area')
        hist_vec = torch.cuda.FloatTensor(b, self.bins_num*3, 4, 4, device=imgs_in.device).fill_(0)

        idx = 1
        for s in range(self.num_scales):
            # 1. downscale image
            if s != 0 :
                h = int(h/self.downscale)
                w = int(w/self.downscale)
                idx*=2
            interval = int(4/idx)

#             # 2. extract histogram 
            for i in range(idx) :
                for j in range(idx) :
                    for k in range(b) : 
                        hists = torch.histc(tgt_imgs[k,:,i*h:(i+1)*h, j*w:(j+1)*w], bins=self.bins_num, min=0, max=1) /(h*w) # assume x is normalize btw 0~1.
                        hist_vec[k, self.bins_num*s:self.bins_num*(s+1), i*interval:(i+1)*interval, j*interval:(j+1)*interval] = hists.view(1,-1, 1, 1).repeat(1, 1, interval, interval)
        return hist_vec

class l2norm(nn.Module):
    def __init__(self):
        super(l2norm,self).__init__()

    def forward(self,input,epsilon = 1e-7):
        assert len(input.size()) == 2,"Input dimension requires 2,but get {}".format(len(input.size()))
        
        norm = torch.norm(input,p = 2,dim = 1,keepdim = True)
        output = torch.div(input,norm+epsilon)
        return output

class ToolSelectNet(nn.Module):
    def __init__(self, pretrained, num_action, bins_num=32):
        super(ToolSelectNet, self).__init__()
        self.feature = FeatureExtractor(bins_num=bins_num)
        self.sobel = kornia.filters.Sobel()
        self.seman_net = alexnet(pretrained)
        self.l2norm = l2norm()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_action),
        )
        self._initialize_weights()

    def forward(self, image):
        img_rgb     = self.bayer_to_rgb(image)

        # semantic branch
        f_sem, _   = self.seman_net(img_rgb)

        # illuminance branch
        img_gray    =   cc.rgb_to_grayscale(img_rgb)
        f_inten     = self.feature(img_gray)

        # grad branch
        img_grad    = self.sobel(img_gray)
        f_grad      = self.feature(img_grad)

        # normalize each feature vector
        b = f_sem.size(0)

        f_sem = f_sem.reshape(b, -1)
        f_inten = f_inten.reshape(b, -1)
        f_grad = f_grad.reshape(b, -1)

        f_sem = self.l2norm(f_sem)
        f_inten = self.l2norm(f_inten)
        f_grad = self.l2norm(f_grad)

        f_cat = torch.cat([f_sem, f_inten, f_grad], dim=1)
        output = self.classifier(f_cat)
        return output

    def bayer_to_rgb(self, inputs):
        B_, C_,H_,W_ = inputs.shape
        inputs_ = torch.cuda.FloatTensor(B_, 3, H_,W_, device=inputs.device)
        inputs_[:, 0,:,:] = inputs[:, 0,:,:]
        inputs_[:, 1,:,:] = (inputs[:, 1,:,:] + inputs[:, 2,:,:])/2
        inputs_[:, 2,:,:] = inputs[:, 3,:,:]
        return inputs_

    def _initialize_weights(self):
        cnt = 0
        for m in self.classifier.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 2:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                    

from torch.autograd import Variable
import numpy as np
if __name__=='__main__':
#    img = torch.empty((3, 3, 224, 224)).normal_(0., 0.01)
    net = ToolSelectNet_ori(pretrained=True, num_action=10).cuda()
    img = torch.rand((30, 3, 63, 63)).cuda()
    idx = 0
    while True: 
#        img = torch.empty((3, 3, 63, 63)).normal_(0., 0.8).cuda()
        output = net(img)
#        time.sleep(1)
        if idx%100 ==0 :
            print(output)   
        idx = idx+1
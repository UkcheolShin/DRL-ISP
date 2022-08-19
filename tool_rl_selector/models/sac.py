import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import kornia.color as cc
from torch.distributions import Categorical
from torchvision import models

from .dqn import CNN_Extractor
from .TSnet import FeatureExtractor, alexnet, l2norm
from .utils import disable_gradients, conv2d_size_out
from reward.utils import bayer_to_rgb


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RandomEncoder(nn.Module):
    def __init__(self):
        super(RandomEncoder, self).__init__()
        self.h = 128
        self.w = 128
        self.conv1 = nn.Conv2d(4, 32, kernel_size=9, padding=9 // 2, stride=2)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 24, kernel_size=5, padding=5 // 2, stride=2)
        # self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        # self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        # self.bn4 = nn.BatchNorm2d(24)

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w, kernel_size=9, padding=9 // 2), padding=5 // 2),
                            padding=5 // 2), padding=5 // 2)
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h, kernel_size=9, padding=9 // 2), padding=5 // 2),
                            padding=5 // 2), padding=5 // 2)
        linear_input_size = convw * convh * 24

        self.linear1 = nn.Linear(linear_input_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        # self.output_size = linear_input_size
        self.output_size = 256

    def forward(self, x):
        x = F.interpolate(x, (self.h, self.w), mode='area')
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        x = F.sigmoid(self.layer_norm(self.linear1(x)))
        return x

    def get_output_size(self):
        return self.output_size


class QNetwork(nn.Module):

    def __init__(self, num_channels, num_actions, num_feature=256, dueling_net=False,
                 layers=2, layer_activation='relu'):
        super().__init__()

        if dueling_net:
            self.a_head = self.create_network(num_channels, num_actions, num_feature, layer_activation, layers)

            self.v_head = self.create_network(num_channels, 1, num_feature, layer_activation, layers)

        else:
            self.head = self.create_network(num_channels, num_actions, num_feature, layer_activation, layers)

        self.dueling_net = dueling_net

    def create_network(self, num_channels, num_actions, num_feature, activation, layers, last_activation='none'):
        networks = []
        for layer in range(layers):
            in_channel = num_feature
            out_channel = num_feature
            if layer == 0:
                in_channel = num_channels
            elif layer == layers - 1:
                out_channel = num_actions

            networks.append(nn.Linear(in_channel, out_channel))
            if layer != layers - 1:
                cur_activation = activation
            else:
                cur_activation = last_activation

            if cur_activation == 'relu':
                networks.append(nn.ReLU())
            elif cur_activation == 'tanh':
                networks.append(nn.Tanh())
            elif cur_activation == 'sigmoid':
                networks.append(nn.Sigmoid())
            elif cur_activation == 'none':
                pass
            else:
                raise NotImplementedError

        return nn.Sequential(*networks)

    def forward(self, states):
        if self.dueling_net:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)
        else:
            return self.head(states)


class SACBase(nn.Module):
    def __init__(self, h, w, num_action, feature_extractor='cnn', pretrained=None, resize_denominator=1,
                 not_use_inten=False, not_use_grad=False, not_use_seman=False, seman_net_name='AlexNet'):
        super(SACBase, self).__init__()
        self.interpolation_enabled = False
        self.not_use_inten = not_use_inten
        self.not_use_grad = not_use_grad
        self.not_use_seman = not_use_seman
        self.seman_net_name = seman_net_name
        if feature_extractor != 'shared':
            self.h = int(h // resize_denominator)
            self.w = int(w // resize_denominator)

            if self.h != h:
                self.interpolation_enabled = True
        else:
            self.h = h
            self.w = w

        if feature_extractor == 'cnn':
            self.feature = CNN_Extractor(self.h, self.w)
            size1 = self.feature.get_output_size()
            self.trainable_feature = True
        elif feature_extractor == 'hist':
            size1 = 0
            if not self.not_use_seman:
                if self.seman_net_name == 'AlexNet':
                    self.seman_net = alexnet(pretrained, input_h=self.h, input_w=self.w)

                    cur_size = self.seman_net.get_output_size()
                elif 'ResNet' in self.seman_net_name:
                    if self.seman_net_name == 'ResNet18':
                        self.seman_net = models.resnet18(pretrained=True)
                    elif self.seman_net_name == 'ResNet34':
                        self.seman_net = models.resnet34(pretrained=True)
                    elif self.seman_net_name == 'ResNet50':
                        self.seman_net = models.resnet50(pretrained=True)
                    elif self.seman_net_name == 'ResNet101':
                        self.seman_net = models.resnet101(pretrained=True)
                    else:
                        raise NotImplementedError
                    self.seman_net.eval()
                    cur_size = self.seman_net.fc.in_features
                    self.seman_net.fc = Identity()
                else:
                    raise NotImplementedError

                disable_gradients(self.seman_net)

                # self.seman_norm = nn.LayerNorm(cur_size)
                size1 += cur_size

            if not self.not_use_grad:
                self.sobel = kornia.filters.Sobel()
            if (not self.not_use_inten) or (not self.not_use_grad):
                self.feature = FeatureExtractor(bins_num=256)
                disable_gradients(self.feature)

            if not self.not_use_grad:
                cur_size = self.feature.get_output_size()
                # self.grad_norm = nn.LayerNorm(cur_size)
                size1 += cur_size

            if not self.not_use_inten:
                cur_size = self.feature.get_output_size()
                # self.inten_norm = nn.LayerNorm(cur_size)
                size1 += cur_size
            # self.all_norm = nn.LayerNorm(size1)
            self.trainable_feature = False
        elif feature_extractor == 'shared':
            size1 = h
            self.trainable_feature = False
        elif feature_extractor == 'rand':
            self.feature = RandomEncoder()
            self.feature.eval()
            disable_gradients(self.feature)
            size1 = self.feature.get_output_size()
            self.trainable_feature = False
        else:
            raise NotImplementedError
        self.feature_size = size1
        self.feature_extractor = feature_extractor
        self.feature_dict = {}
        self.l2norm = l2norm()

    def forward_feature_test(self, x):
        if not self.trainable_feature:
            split_x = x.split(1, dim=0)
            id_x = [[idx, id(x)] for idx, x in enumerate(split_x)]
            in_feature = [x for x in id_x if x[1] in self.feature_dict]
            if len(in_feature) != 0:
                if len(in_feature) == x.shape[0]:
                    f_cat = torch.cat([self.feature_dict[x[1]] for x in id_x], dim=0)
                    return f_cat

                not_in_feature = [x for x in id_x if x[1] not in self.feature_dict]
                not_in_index = [x[0] for x in not_in_feature]
                x = torch.cat([split_x[x[0]] for x in not_in_feature], dim=0)
            else:
                not_in_index = [idx for idx in range(x.shape[0])]

        if self.interpolation_enabled:
            x = F.interpolate(x, (self.h, self.w), mode='area')
        if self.feature_extractor == 'cnn':
            f_cat = self.feature(x)
        elif self.feature_extractor == 'hist':
            _, C_, _, _ = x.shape
            if C_ == 4:
                x = bayer_to_rgb(x)
            
            with torch.no_grad():
                # semantic branch
                features = []
                if not self.not_use_seman:
                    f_sem, _ = self.seman_net(x)
                    b = f_sem.size(0)
                    f_sem.reshape(b, -1)
                    f_sem = self.l2norm(f_sem)
                    # f_sem = self.seman_norm(f_sem)
                    features.append(f_sem)

                # illuminance branch
                if (not self.not_use_inten) or (not self.not_use_grad):
                    img_gray = cc.rgb_to_grayscale(x)

                if not self.not_use_inten:
                    f_inten = self.feature(img_gray)
                    b = f_inten.size(0)
                    f_inten.reshape(b, -1)
                    f_inten = self.l2norm(f_inten)
                    # f_inten = self.inten_norm(f_inten)
                    features.append(f_inten)

                # grad branch
                if not self.not_use_grad:
                    img_grad = self.sobel(img_gray)
                    f_grad = self.feature(img_grad)
                    b = f_grad.size(0)
                    f_grad.reshape(b, -1)
                    f_grad = self.l2norm(f_grad)
                    # f_grad = self.grad_norm(f_grad)
                    features.append(f_grad)

                f_cat = torch.cat(features, dim=1)
                # f_cat = self.all_norm(f_cat)

        elif self.feature_extractor == 'rand':
            f_cat = self.feature(x)
        elif self.feature_extractor == 'shared':
            f_cat = x
        else:
            raise NotImplementedError
        if not self.trainable_feature:
            feature_split = f_cat.split(1, dim=0)
            not_in_feature_id = [id_x[x][1] for x in not_in_index]
            for f1, id1 in zip(feature_split, not_in_feature_id):
                self.feature_dict[id1] = f1
            if len(in_feature) != 0:
                f_cat = torch.cat([self.feature_dict[x[1]] for x in id_x])
        return f_cat

    def forward_feature(self, x):
        if self.interpolation_enabled:
            x = F.interpolate(x, (self.h, self.w), mode='area')
        if self.feature_extractor == 'cnn':
            f_cat = self.feature(x)
        elif self.feature_extractor == 'hist':
            _, C_, _, _ = x.shape
            if C_ == 4:
                x = bayer_to_rgb(x)

            with torch.no_grad():
                # semantic branch
                features = []
                if not self.not_use_seman:
                    if 'ResNet' in self.seman_net_name:
                        f_sem = self.seman_net(x)
                    else:
                        f_sem, _ = self.seman_net(x)
                    b = f_sem.size(0)
                    f_sem = f_sem.reshape(b, -1)
                    f_sem = self.l2norm(f_sem)
                    features.append(f_sem)

                # illuminance branch
                if (not self.not_use_inten) or (not self.not_use_grad):
                    img_gray = cc.rgb_to_grayscale(x)

                if not self.not_use_inten:
                    f_inten = self.feature(img_gray)
                    b = f_inten.size(0)
                    f_inten = f_inten.reshape(b, -1)
                    f_inten = self.l2norm(f_inten)
                    features.append(f_inten)

                # grad branch
                if not self.not_use_grad:
                    img_grad = self.sobel(img_gray)
                    f_grad = self.feature(img_grad)
                    b = f_grad.size(0)
                    f_grad = f_grad.reshape(b, -1)
                    f_grad = self.l2norm(f_grad)
                    features.append(f_grad)

                f_cat = torch.cat(features, dim=1)

        elif self.feature_extractor == 'shared':
            f_cat = x
        elif self.feature_extractor == 'rand':
            f_cat = self.feature(x)
        else:
            raise NotImplementedError
        return f_cat


class SACQNetwork(SACBase):
    def __init__(self, h, w, num_action, feature_extractor='cnn', pretrained=None, resize_denominator=1,
                 layers=2, layer_hidden=256, layer_activation='relu'):
        super(SACQNetwork, self).__init__(h, w, num_action, feature_extractor, pretrained, resize_denominator)
        self.q1 = QNetwork(self.feature_size, num_action,
                           dueling_net=True, layers=layers, layer_activation=layer_activation, num_feature=layer_hidden)
        self.q2 = QNetwork(self.feature_size, num_action,
                           dueling_net=True, layers=layers, layer_activation=layer_activation, num_feature=layer_hidden)

    def forward(self, states):
        x = self.forward_feature(states)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def get_parameters(self):
        if self.feature_extractor == 'cnn':
            return itertools.chain(self.q1.parameters(), self.q2.parameters(), self.feature.parameters())
        else:
            return itertools.chain(self.q1.parameters(), self.q2.parameters())

    def get_named_parameters(self):
        if self.feature_extractor == 'cnn':
            return itertools.chain(self.q1.named_parameters(), self.q2.named_parameters(), self.feature.named_parameters())
        else:
            return itertools.chain(self.q1.named_parameters(), self.q2.named_parameters())


class SACPolicy(SACBase):
    def __init__(self, h, w, num_action, feature_extractor='cnn', pretrained=None, resize_denominator=1,
                 layers=2, layer_hidden=256, layer_activation='relu',
                 not_use_inten=False, not_use_grad=False, not_use_seman=False, seman_net_name="AlexNet"):
        super(SACPolicy, self).__init__(h, w, num_action, feature_extractor, pretrained, resize_denominator,
                                        not_use_inten, not_use_grad, not_use_seman, seman_net_name=seman_net_name)
        self.head = QNetwork(self.feature_size, num_action,
                             dueling_net=False, layers=layers, layer_activation=layer_activation, num_feature=layer_hidden)

    def select_action(self, states):
        x = self.forward_feature(states)
        action_logits = self.head(x)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return x, action_logits, greedy_actions

    def forward_without_feature(self, x):
        action_logits = self.head(x)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return x, actions, action_probs, log_action_probs

    def forward(self, states):
        x = self.forward_feature(states)
        return self.forward_without_feature(x)

    def get_parameters(self):
        if self.feature_extractor == 'cnn':
            return itertools.chain(self.feature.parameters(), self.head.parameters())
        else:
            return self.head.parameters()

    def get_named_parameters(self):
        if self.feature_extractor == 'cnn':
            return itertools.chain(self.feature.named_parameters(), self.head.named_parameters())
        else:
            return self.head.named_parameters()


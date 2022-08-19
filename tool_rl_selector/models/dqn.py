import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import conv2d_size_out

class DQN(nn.Module):
    def __init__(self, h, w, num_action):
        super(DQN, self).__init__()
        self.h = 112; self.w = 112
        self.conv1 = nn.Conv2d(4, 32, kernel_size=9, padding=9 // 2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn4 = nn.BatchNorm2d(24)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w, kernel_size=9, padding=9//2), padding=5//2), padding=5//2), padding=5//2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h, kernel_size=9, padding=9//2), padding=5//2), padding=5//2), padding=5//2)
        linear_input_size = convw * convh * 24
        self.fcn1 = nn.Linear(linear_input_size, 256)
        self.q_action = nn.Linear(256, num_action)

    def get_output_size(self):
        return self.linear_input_size

    def forward(self, x):
        x = F.interpolate(x, (self.h, self.w), mode='area')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        x_vec = F.relu(self.fcn1(x))
 
        action = self.q_action(x_vec) # batch x action size
#        action_max = torch.argmax(action, dim=1)
        return action

class CNN_Extractor(nn.Module):
    def __init__(self, h, w):
        super(CNN_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=9, padding=9 // 2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5, padding=5 // 2, stride=2)
        self.bn4 = nn.BatchNorm2d(24)
        self.h = h
        self.w = w

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w, kernel_size=9, padding=9//2), padding=5//2), padding=5//2), padding=5//2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h, kernel_size=9, padding=9//2), padding=5//2), padding=5//2), padding=5//2)
        linear_input_size = convw * convh * 24

    def get_output_size(self):
        return self.linear_input_size

    def forward(self, x):
        x = F.interpolate(x, (self.h, self.w), mode='area')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        return x

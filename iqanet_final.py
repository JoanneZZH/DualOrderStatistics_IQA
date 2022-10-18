import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torch.autograd import Function

from iqanet_init import IQANet_init


class IQANet_DGap(nn.Module):
    def __init__(self):
        super(IQANet_DGap, self).__init__()

        resnet101 = models.resnet101(True)
        self.resnet101_freeze = nn.Sequential(*list(resnet101.children())[:7])
        self.resnet101 = nn.Sequential(*list(resnet101.children())[7:8])

        self.resnet_mytrain = IQANet_init()
        
        print("=> using pre-trained model.")
        arch = '20221011-172658_koniq10k_GAP_fc_train_RTX'
        # arch = '20210627-215044LIVEV_GAP-fc_train_RTX20epochs_best'
        # arch = '20210629-174833kadid10k_GAP_fc_train40epochs'
        path = 'checkpoints/{}.pth.tar'.format(arch)
        state_dict = torch.load(path, map_location='cpu')['state_dict']
        self.resnet_mytrain.load_state_dict(state_dict)

        self.myResnet_freeze = nn.Sequential(*list(self.resnet_mytrain.children())[:2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))

        self.fc1 = list(self.resnet_mytrain.children())[3]
        self.fc2 = nn.Linear(2048,1)

        self.param_w_shape = self.fc2.weight.shape
        self.param_b_shape = self.fc2.bias.shape
        self.fc2.weight.data = torch.from_numpy(np.zeros((self.param_w_shape))).to(torch.float32).cuda()
        self.fc2.bias.data = torch.from_numpy(np.zeros((self.param_b_shape))).to(torch.float32).cuda()
        self.fc = nn.Linear(2,1)  # Need modification for distribution scores in Koniq10k database

        # freeze conv and weight of batchnorm
        for para in (self.myResnet_freeze.parameters() or self.resnet101_freeze.parameters()):
            para.requires_grad = False

        # freeze running mean and var of batchnorm
        self.myResnet_freeze.eval()
        self.resnet101_freeze.eval()

    def forward(self, x):

        x0 = self.myResnet_freeze(x)
        x1 = self.avgpool(x0)

        x2 = self.resnet101_freeze(x)
        x2 = self.resnet101(x2)
        x2 = self.avgpool(x2)

        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        x2 = x2.view(x2.size(0), -1)
        x4 = self.fc2(x2)

        x = torch.cat([x1, x4],1) 
        x = self.fc(x)

        return x

    def train(self, mode=True):
        self.training = mode

        for m in [self.resnet101, self.fc1, self.fc2, self.fc]:
            m.training = mode
            for module in m.children():
                module.train(mode)

        return self

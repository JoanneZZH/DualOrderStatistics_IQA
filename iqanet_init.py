import torch
import torch.nn as nn
import torchvision.models as models



class IQANet_init(nn.Module):
    def __init__(self):
        super(IQANet_init, self).__init__()

        resnet101 = models.resnet101(True)

        self.resnet101_freeze = nn.Sequential(*list(resnet101.children())[:7])
        self.resnet101 = nn.Sequential(*list(resnet101.children())[7:8])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))

        fc_features = resnet101.fc.in_features
        # self.fc = nn.Linear(2048, 5)  # for Koniq10K wieh 5 distribution value

        # for other database with one value target
        self.fc = nn.Linear(fc_features, 1)  

        # freeze conv and weight of batchnorm
        for para in self.resnet101_freeze.parameters():
            para.requires_grad = False

        # freeze running mean and var of batchnorm
        self.resnet101_freeze.eval()

    def forward(self, x):
        x = self.resnet101_freeze(x)
        x = self.resnet101(x)
        x = self.avgpool(x)
        x= x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train(self, mode=True):
        self.training = mode

        for m in [self.resnet101, self.fc]:
            m.training = mode
            for module in m.children():
                module.train(mode)

        return self

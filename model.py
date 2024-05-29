import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tm
import numpy as np


class ResFCN(nn.Module):
    def __init__(self):
        super(ResFCN,self).__init__()
        self.resnet = tm.resnet50(pretrained=True)
        self.bn = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.score_32s = nn.Conv2d(512 * 4,
                                   256,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256*4 ,
                                   256,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 * 4,
                                  256,
                                  kernel_size=1)
        self.tmconv = nn.Conv2d(256,
                                  1,
                                  kernel_size=1)

    def forward(self, img):
        img = self.features1(img)
        img = self.resnet.maxpool(img)
        img = self.resnet.layer1(img)
        img = self.resnet.layer2(img)
        img_8 = self.score_8s(img)

        img = self.resnet.layer3(img)

        img_16 = self.score_16s(img)

        img = self.resnet.layer4(img)

        img_32 = self.score_32s(img)

        img_16_spatial_dim = img_16.size()[2:]
        img_8_spatial_dim = img_8.size()[2:]
        img_16 += nn.functional.interpolate(img_32,
                                                size=img_16_spatial_dim,
                                                mode="bilinear",
                                                align_corners=True)

        img_8 += nn.functional.interpolate(img_16,
                                               size=img_8_spatial_dim,
                                               mode="bilinear",
                                               align_corners=True)

        img_upsampled = nn.functional.interpolate(img_8,
                                                     size=(224,224),
                                                     mode="bilinear",
                                                     align_corners=True)


        img_upsampled = img_upsampled
        img_upsampled = self.tmconv(img_upsampled)
        img_upsampled = torch.sigmoid(img_upsampled).squeeze()
        return img_upsampled
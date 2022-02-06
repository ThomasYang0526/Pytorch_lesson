#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:49:55 2022

@author: SuTungYang
"""

import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false_1 = torchvision.models.vgg16(pretrained=False)
vgg16_false_2 = torchvision.models.vgg16(pretrained=False)
# vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

print(vgg16_false_1)
vgg16_false_1.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_false_1)

print(vgg16_false_2)
vgg16_false_2.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false_2)
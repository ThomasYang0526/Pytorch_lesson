#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 18:51:42 2022

@author: SuTungYang
"""

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = MyModel()
print(model)
input = torch.ones((64, 3, 32, 32))
output = model(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(model, input)
writer.close()
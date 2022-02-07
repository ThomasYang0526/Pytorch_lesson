#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:43:31 2022

@author: SuTungYang
"""

import torch

x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
print(x.grad)
z = x.pow(2).sum()
z.backward()
print(x.grad)

optimizer = torch.optim.SGD([x], 0.1)
optimizer.zero_grad()
print(x.grad)
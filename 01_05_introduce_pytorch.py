#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:54:27 2022

@author: SuTungYang
"""

import torch

print(torch.cuda.is_available())
print("="*80)
print(dir(torch))
print("="*80)
print(dir(torch.cuda))
print("="*80)
help(torch.cuda.is_available)

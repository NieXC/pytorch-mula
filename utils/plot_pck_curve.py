import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from utils.calc_pckh import pck_table_output_lip_dataset 

path = 'exps/snapshots/pil_lip.pth.tar'

checkpoint = torch.load(path)
pck_avg_list = checkpoint['pck_avg_list']
pck_all_list = checkpoint['pck_all_list']

for ei in range(0, len(pck_all_list)):
    print('Epoch: [{0}] =============='.format(ei))
    pck_all = pck_all_list[ei]
    pck_table_output_lip_dataset(pck_all[-1], 'Ours')

print('Current epoch: {0}'.format(checkpoint['epoch']))

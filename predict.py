import numpy as np
import argparse
from utils import load_images
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PTModel, WPNet

parser = argparse.ArgumentParser(description='WPNet')
parser.add_argument('--model', default='/models/nyu.pt', type=str, help='Trained Monodepth Pytorch model file.')
parser.add_argument('--input', default='my_examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Goal situation : End to End learning

model = PTModel().float()
model.load_state_dict(torch.load(args.model))
# model.eval()
wpnet = WPNet(model)








import os
import sys
import glob
import argparse
import matplotlib
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0,"../")
sys.path.insert(1,"./")
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
#from layers import BilinearUpSampling2D
from utils import load_images
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from pytorch_model_skip import PTModel as Model

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='./model_2.pth', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='/home/arunava/dense_depth/test_0/images/img__0_1548623204601919700.png', type=str, help='Input filename or folder.')
args = parser.parse_args()
print(args)


def my_DepthNorm(x, maxDepth):
    return maxDepth / x

def my_predict( images, minDepth=10, maxDepth=1000):
    model = Model().cuda()
    model_dict=torch.load("model_2.pth")
    model.load_state_dict(model_dict)

    with torch.no_grad():
    # Compute predictions
        predictions = model(images)

    # Put in expected range
    return np.clip(my_DepthNorm(predictions.cpu().numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def show(args):
  # # Input images
  print('imagefile before',args.input)
  # inputs = load_images( glob.glob(args.input) ).astype('float32')
  inputs = load_images( [args.input]).astype('float32')

  pytorch_input = torch.from_numpy(inputs[0,:,:,:]).permute(2,0,1).unsqueeze(0).to("cuda")
  print(type(pytorch_input))
  # print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

  # # Compute results
  output = my_predict(pytorch_input)
  print(output.shape)
  cv2.imshow("Test", output[0][0])
  cv2.imwrite('test.png',output[0][0])
  cv2.waitKey(0)

show(args)

import os
import glob
import time
import argparse
import numpy as np
from io import BytesIO

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


from pytorch_model_skip_kanade import PTModel as Model

from utils_np import predict, load_images, display_images, evaluate, DepthNorm
from matplotlib import pyplot as plt
import data as data_load

from data import loadZipToMem, getNoTransform
from torch.utils.data import Dataset, DataLoader
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
# custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

def eval():
    # Load model into GPU / CPU
    print('Loading model...')
    # model = load_model(args.model, custom_objects=custom_objects, compile=False)
    model = Model().cuda()
    e=[]
    for i in os.listdir():
        if 'model' in i and '.pth' in i:
            e.append(i)
    e=sorted(e)[-1]
    print("loading "+e+" for eval...")
    model_dict=torch.load(e)
    model.load_state_dict(model_dict)

    # Load test data
    print('Loading test data...', end='')

    import numpy as np
    from data import extract_zip
    data = extract_zip('nyu_test.zip')
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy'])).astype('float32')
    depth = np.load(BytesIO(data['eigen_test_depth.npy'])).astype('float32')
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')

    start = time.time()
    print('Testing...')

    e = evaluate(model, rgb, depth, crop, batch_size=6)

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    end = time.time()
    print('\nTest time', end-start, 's')


if __name__ == '__main__':
    eval()
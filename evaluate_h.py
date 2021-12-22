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

from pytorch_model_skip import PTModel as Model

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
#custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
model = Model().cuda()
model_dict=torch.load("model_2.pth")
model.load_state_dict(model_dict)

# Load model into GPU / CPU
print('Loading model...')
#model = load_model(args.model, custom_objects=custom_objects, compile=False)

# Load test data
print('Loading test data...')

def getTrainingTestingData(batch_size):
    data, nyu2_test = loadZipToMem('nyu_data.zip')

    transformed_testing = data_load.depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_testing, batch_size, shuffle=False)

test_loader=getTrainingTestingData(8)

with torch.no_grad():
    y_test=np.empty((1,1,240,320))

    for i, sample_batched in enumerate(test_loader):
        print("Data : ",i)
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        
        # Normalize depth
        depth_n = DepthNorm( depth )

        # Predict
        output = model(image)
        # print(output.cpu().shape,depth.cpu().shape)
        y_test=np.vstack((y_test,np.vstack((output.cpu(),depth.cpu()))))
        
start = time.time()
print('Testing...')
def get_crop_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO

    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print(crop[0])
    print('Test data loaded.\n')
    return crop
e = evaluate( y_test,get_crop_test_data(), batch_size=6)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

end = time.time()
print('\nTest time', end-start, 's')

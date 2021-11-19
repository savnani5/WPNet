import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGBA2RGB)
        x = np.clip(np.asarray(x, dtype=float)/255, 0, 1)
        # x = np.clip(np.asarray(Image.open( file ), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def my_DepthNorm(x, maxDepth):
    return maxDepth / x

def my_predict(model, images, minDepth=10, maxDepth=1000):

    with torch.no_grad():
        # Compute predictions
        predictions = model(images)

        # Put in expected range
    return np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

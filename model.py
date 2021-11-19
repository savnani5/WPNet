import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from utils import my_DepthNorm, my_predict

# ______________________________________Monodepth Autoencoder Network__________________________________________________

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( pretrained=False )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

# __________________________________________Downsampling Block_____________________________________________________

class WPNet(nn.Module):
    """Waypoint Prediction Network"""
    
    def __init__(self, input_shape, monodepth_model, lr=0.001):
        
        # Don't define flatten and Relu in constructor
        super(WPNet, self).__init__()       
        self.monodepth_model = monodepth_model # change
        
        # Layer freezing -- change later to unfreeze some layers - currently all layers freezed
        for param in self.monodepth_model.parameters():
            param.requires_grad = False
        
        self.input_shape = input_shape             #input batch shape
        self.conv_net = torch.nn.Sequential(
                        nn.Conv2d(1, 16, 3),       #tweak its input
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(16, 32, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(32, 64, 3),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))
        
        # Hack to find the shape to input to flatten layer
        with torch.no_grad():
            dummy = torch.zeros((input_shape))
            x = self.monodepth_model(dummy)
            x = self.conv_net(x)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]

        self.fc_net = torch.nn.Sequential(
                        nn.Linear(fc_size, 500),
                        nn.Dropout(p=0.2),
                        nn.Linear(500, 100),
                        nn.Linear(100, 20),
                        nn.Linear(20, 7))

        # self.fc3 = nn.Linear(20, 3)
        # self.fc4 = nn.Linear(20, 4)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    # Seperation of carts and quats maybe due to different scaling - explore later - also remove last layer from fc_net
    # def cartesian_forward(self, input_batch):
    #     """Downsampling block to predict x,y,z waypoints"""
    #     x = self.monodepth_model(input_batch)
    #     x = self.conv_net(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc_net(x)
    #     output = self.fc3(x)
    #     return output

    # def quaternion_forward(self, input_batch):
    #     """Downsampling block to predict rotational state in qw, qx, qy, qz quaternions"""
    #     x = self.monodepth_model(input_batch)
    #     x = self.conv_net(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc_net(x)
    #     output = self.fc4(x)
    #     return output


    def forward(self, input_batch):
        """Returns a vector of waypoints and quarternions - not sure if I wanna return that for n frames or not ??"""
        x = self.monodepth_model(input_batch)
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        output = self.fc_net(x)
        return output

# _____________________________________________________________________________________________________________________


if __name__ == "__main__":

    tensor = torch.zeros((1, 3, 432, 768))
    model = PTModel().float()
    model.load_state_dict(torch.load("models/nyu.pt"))
    wpnet = WPNet(tuple(tensor.shape), model)
    # print(wpnet.cartesian_forward(tensor))
    # print(wpnet.quaternion_forward(tensor))
    print(wpnet.forward(tensor))

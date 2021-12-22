import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


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
        
class res_block(nn.Sequential):
    def __init__(self, in_channels, out_channels,stride=1):
        super(res_block, self).__init__()        
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        # print('middle res block',x.size())
        out = self.block(x)
        # print('after res block',out.size())
        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        # print('after res block',out.size())
        out = F.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)
        
        self.skip11=res_block(in_channels=64, out_channels=features//8)
        self.skip12=res_block(in_channels=features//8, out_channels=features//4)
        self.skip13=res_block(in_channels=features//4, out_channels=64)
        self.skip21=res_block(in_channels=64, out_channels=features//4)
        self.skip22=res_block(in_channels=features//4, out_channels=64)
        self.skip31=res_block(in_channels=128, out_channels=128)
        self.skip41=res_block(in_channels=256, out_channels=256)
        
        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))
        
        # print('x_d0',x_d0.size())
        # print('x_block4',x_block4.size())
        # print('x_block3',x_block3.size())
        # print('x_block2',x_block2.size())
        # print('x_block1',x_block1.size())
        # print('x_block0',x_block0.size())
        
        x_skip4=self.skip41(x_block3)
        # print('x_skip4',x_skip4.size())

        x_d1 = self.up1(x_d0, x_skip4)
        # print('x_d1',x_d1.size())
        
        x_skip3=self.skip31(x_block2)
        x_d2 = self.up2(x_d1, x_skip3)
        # print(x_d2.size())
        
        x_skip2=self.skip22(self.skip21(x_block1))
        x_d3 = self.up3(x_d2, x_skip2)
        # print(x_d3.size())
        
        x_skip1=self.skip13(self.skip12(self.skip11(x_block0)))
        x_d4 = self.up4(x_d3, x_skip1)
        # print(x_d4.size())
        
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( pretrained=True )
        for params in self.original_model.parameters():
            params.requires_grad = False
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


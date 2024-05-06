import torch
import random
import math
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import time

__all__ = ['MC_UNet', 'LRL' ]


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act = nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out

class VGGBlock_MC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x, train=True):
        out = self.conv1(F.dropout3d(x, training=train, p=0.3))
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out 
 
 
class MC_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
 
        nb_filter = [32, 64, 96, 192, 384] 

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
 
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock_MC(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock_MC(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock_MC(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock_MC(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, 1, padding=0)

        for m in self.modules():
          if isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
               init.zeros_(m.bias)
          elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, input, train=True, T=1):


        x0_0 = self.conv0_0(input)     
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
 
        output_final = None 

        for i in range(0, T):
 
          x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
          x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
          x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
          x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
     
          output = self.final(x0_4)
        
          if T <= 1: 
            output_final = output     
          else:
            if i == 0:
               output_final = output.unsqueeze(0)
            else:
               output_final = torch.cat((output_final, output.unsqueeze(0)), dim=0)
   
        return output_final

class LRL(nn.Module):
      def __init__(self, num_classes, input_channels=3, resize_w = 128, resize_h = 128, **kwargs):
        super().__init__()

        self.nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.up = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv0_0 = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv4_0 = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4], nn.ReLU(inplace=True))
        
        self.conv_down_ = nn.Conv3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_up_ = nn.ConvTranspose3d(self.nb_filter[4], self.nb_filter[4], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv3_1 = VGGBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], self.nb_filter[3], nn.ReLU(inplace=True))
        self.conv2_2 = VGGBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2], nn.ReLU(inplace=True))
        self.conv1_3 = VGGBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1], nn.ReLU(inplace=True))
        
        self.conv0_4 = VGGBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0], nn.ReLU(inplace=True))

        self.final = nn.Conv3d(self.nb_filter[0], num_classes, kernel_size=1)
       
        for m in self.modules():
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
               init.zeros_(m.bias) 

      def add_noise_chw(self, x, prob = 0.05):
        b, c, d, h, w = x.shape

        flat_x = x.reshape(b,c*d*h*w)
        noise_x = flat_x.clone()

        random_change = torch.bernoulli(torch.full((c*d*h*w,), 1-prob, device='cuda'))
        random_num = torch.rand((c*d*h*w,),device='cuda')
        random_num[random_change.bool()]=0
        noise_x += random_num
 
        return noise_x.reshape(b,c,d,h,w)

      def forward(self, input, M=1, noise_branch = False, feature_dis = False):

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_ = self.conv_down_(x4_0)

        x4_0 = self.conv_up_(x4_0_)
        x3_1 = self.conv3_1(torch.cat([x3_0.repeat(M, 1, 1, 1, 1), self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0.repeat(M, 1, 1, 1, 1), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0.repeat(M, 1, 1, 1, 1), self.up(x2_2)], 1))
        
        x0_4 = self.conv0_4(torch.cat([x0_0.repeat(M, 1, 1, 1, 1), self.up(x1_3)], 1))

        output = self.final(x0_4)

        if noise_branch:
          x1_3_noise = self.add_noise_chw(x1_3)
          x0_4_noise = self.conv0_4(torch.cat([x0_0.repeat(M, 1, 1, 1, 1), self.up(x1_3_noise)], 1))

          output_noise = self.final(x0_4_noise)
          return output, output_noise
        else:
          return output

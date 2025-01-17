import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
__all__ = ['BCEDiceLoss', 'BceLoss', 'DiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target): 
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='sum')/(input.size(-1)*input.size(-2)*input.size(-3))
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input = input.view(num, -1)
        target = target.view(num, -1)

        intersection = (input * target) 

        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() +smooth)
        dice = 1.0 - dice

        return   bce + 2 * dice


class DiceLoss(nn.Module):
      def __init__(self):   
        super().__init__()
 
      def forward(self, input, target): 

        smooth = 1e-5 
        input = torch.softmax(input, dim=1)
        num = target.size(0)

        input = input[:,1:,:,:,:]
        target = target[:,1:,:,:,:]

        input = input.contiguous().view(num, -1)
        target = target.contiguous().view(num, -1)  

        intersection = (input * target) 

        dice = (2 * intersection.sum(dim=1) + smooth) / (input.sum(dim=1) + target.sum(dim=1) +smooth)
        all_dice = 1.0 - dice

        return  all_dice.sum()

class CELoss(nn.Module):
      def __init__(self,num_classes = 4):   
        super().__init__()
        self.num_classes = num_classes
 
      def forward(self, input, target): 
        num = input.size(0)
        depth = input.size(2)
        w = input.size(-2)
        h = input.size(-1) 
        target = (target > 0.5).int()#[Batch,C,Depth,Width,Height]
       
        target_n = None
        for i in range(self.num_classes):#one_hot->corresponding label
           if target_n is None:
              target_n =  i * target[:, i, :, :, :].unsqueeze(1)
           else:
              target_n = target_n + i * target[:, i, :, :, :].unsqueeze(1)

        input = input.contiguous().view(num*depth, self.num_classes, -1)
        target = (target_n.contiguous().view(num*depth, -1)).long()
 
     
        bce = F.cross_entropy(input, target, reduction='none')
        all_bce = torch.sum(bce) / (w * h * depth)
 
        return all_bce

class BceLossThres(nn.Module):
      def __init__(self,threshold = 0.7):   
        super().__init__()
        self.threshold = threshold
 
      def forward(self, input, target, num_classes=2): 
        depth = input.size(2)
        num = input.size(0)
        w = input.size(-2)
        h = input.size(-1) 
        target = (target > self.threshold).int()#[Batch,C,Depth,Width,Height]
        mask = torch.sum(target,dim = 1).unsqueeze(dim = 1)#[Batch,1,Depth,Width,Height]

        target_n = None
        for i in range(num_classes):
           if target_n is None:
              target_n =  i * target[:, i, :, :, :].unsqueeze(1)
           else:
              target_n = target_n + i * target[:, i, :, :, :].unsqueeze(1)
        target_n += (1-mask) * (-1)

        input = input.contiguous().view(num*depth, num_classes, -1)
        target = (target_n.contiguous().view(num*depth, -1)).long()
 
        bce = F.cross_entropy(input, target, ignore_index=-1, reduction='none')
        all_bce = torch.sum(bce) / (w * h * depth)
 
        return all_bce
 
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import random
from torchvision.utils import save_image
import sys
import cv2

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""
# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.detail = []

    def update(self, val, n=1):
        self.detail.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_single_case(net, img, stride_xy, stride_z, patch_size, num_classes=1):
    image = img.cpu().numpy()
    b, c, d, w, h = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
 

    wl_pad, wr_pad = int(w_pad//2), int(w_pad-w_pad//2)
    hl_pad, hr_pad = int(h_pad//2), int(h_pad-h_pad//2)
    dl_pad, dr_pad = int(d_pad//2), int(d_pad-d_pad//2)
 

    if add_pad:
        image = np.pad(image, [(0, 0), (0, 0),  (dl_pad, dr_pad), (wl_pad,wr_pad), (hl_pad,hr_pad)], mode='constant', constant_values=0)
    bb, cc, dd, ww, hh = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
   
    score_map = np.zeros((bb, num_classes, dd, ww, hh)).astype(np.float32)
    cnt = np.zeros((1, 1, dd, ww, hh)).astype(np.float32)
    out_map = np.zeros((bb, num_classes, dd, ww, hh)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] 
                test_patch = torch.from_numpy(test_patch).cuda()
                

                y1 = net(test_patch, T=10)
                y = torch.softmax(y1, dim=2).mean(0)


                y = y.cpu().data.numpy()
                y1 = y1.cpu().data.numpy() 
                score_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = score_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + y

                cnt[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = cnt[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + 1

                out_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = out_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + y1.mean(0)

    score_map = score_map/cnt
    out_map = out_map/cnt 
    if add_pad:
        out_map = out_map[:, :, dl_pad:dl_pad+d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        score_map = score_map[:, :, dl_pad:dl_pad+d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
 
    return torch.from_numpy(out_map).cuda(), torch.from_numpy(score_map).cuda()

def save_binary_seg(pred_img, save_path, dice_slice, interval=1, num_classes=2):
    #pred_img.shape: [batch=1,class,depth,height,width]
    pred_img = torch.transpose(pred_img, 1, 2)
    pred_img = pred_img.squeeze(dim = 0)#[depth,class,height,width]

    #import pdb; pdb.set_trace()
    for i in range(0,pred_img.size(0),interval):       
        pred_img_slice = (pred_img[i,1,:,:].unsqueeze(dim = 0) > 0.5).int()
        pred_img_slice = pred_img_slice.cpu().numpy()
        pred_img_slice = pred_img_slice.transpose(1,2,0)

        binary_mask = np.array(pred_img_slice, dtype=np.uint8)*255

        cv2.imwrite(save_path + '/slice_%d'%i + '_pred_%.2f'%dice_slice[i] +'.png', binary_mask)

def save_seg_img(img, pred_img, save_path, dice_slice=None, interval = 1, num_classes = 2):

    #img.shape: [batch,depth,channel,height,width]
    #pred_img.shape: [batch,class,depth,height,width]
    img = img.squeeze(dim = 0)#[depth,channel,height,width]
    pred_img = torch.transpose(pred_img, 1, 2)
    pred_img = pred_img.squeeze(dim = 0)#[depth,class,height,width]
    #import pdb;pdb.set_trace()
    color = [0,0,255]#red
    
    for i in range(0,img.size(0),interval):
        img_slice = img[i,:,:,:].cpu().numpy()
        img_slice = (img_slice - np.min(img_slice))/(np.max(img_slice) - np.min(img_slice)) * 255
       
        pred_img_slice = (pred_img[i,1,:,:].unsqueeze(dim = 0) > 0.9).int()
        pred_img_slice = pred_img_slice.cpu().numpy()

        img_slice = img_slice.transpose(1,2,0)
        pred_img_slice = pred_img_slice.transpose(1,2,0)

        img_slice = np.where(pred_img_slice==1, 0.5*img_slice + 0.5*np.full_like(img_slice, color), img_slice)
        cv2.imwrite(save_path + '/slice_%d'%i + '_pred_%.2f'%dice_slice[i] +'.png', img_slice)
        #cv2.imwrite(save_path + '/slice_%d'%i + '_gt' +'.png', img_slice)
    
def save_seg_img_post(img, pred_img, save_path, kind, color, interval = 1, num_classes = 2):

    #print(img.shape)#[batch,depth,channel,height,width]
    #print(pred_img.shape)#[batch,depth,class,height,width]
    img = img.squeeze(dim = 0)#[depth,channel,height,width]
    pred_img = pred_img.squeeze(dim = 0)#[depth,class,height,width]
    
    for i in range(3,img.size(0)-3,interval):
        img_slice = img[i,:,:,:].cpu().numpy()
        img_slice = (img_slice - np.min(img_slice))/(np.max(img_slice) - np.min(img_slice)) * 255
       
        pred_img_slice = (pred_img[i,1,:,:].unsqueeze(dim = 0) > 0.9).int()
        pred_img_slice = pred_img_slice.cpu().numpy()

        img_slice = img_slice.transpose(1,2,0)
        pred_img_slice = pred_img_slice.transpose(1,2,0)

        img_slice = np.where(pred_img_slice==1, np.full_like(img_slice, color), img_slice)
        cv2.imwrite(save_path + '/%d'%(i//interval) + kind +'.png', img_slice)

# def area_connection(result, n_class,area_threshold)
# 	"""
# 	result:预测影像
# 	area_threshold：最小连通尺寸，小于该尺寸的都删掉
# 	"""
# 	#result = to_categorical(result, num_classes=n_class, dtype='uint8')  # 转为one-hot
# 	for i in tqdm(range(n_class)):
# 		# 去除小物体
# 		result[:, :, i] = skimage.morphology.remove_small_objects(result[:, :, i] == 1, min_size=area_threshold, connectivity=1, in_place=True) 
# 		# 去除孔洞
# 		result[:, :, i] = skimage.morphology.remove_small_holes(result[:, :, i] == 1, area_threshold=area_threshold, connectivity=1, in_place=True) 
# 	# 获取最终label
# 	result = np.argmax(result, axis=2).astype(np.uint8)
	
# 	return result


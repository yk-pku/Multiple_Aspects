import argparse
import os
from collections import OrderedDict 

import pandas as pd 
import torch
 
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np  
import torchvision.transforms as transforms 
  
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import  save_checkpoint  

import archs
import losses
from dataset import Dataset
from metrics import dice_coef
from utils import AverageMeter, sigmoid_rampup

import sys
import time

from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='AtriaSeg_16',
                        help='experiment name')
    parser.add_argument('--model_save_dir', default='./results')
    parser.add_argument('--epochs', default=160, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size. Note that this is the batch size of patients rather than CT slices')

    # model
    parser.add_argument('--archG', metavar='ARCH', default='LRL',
                        choices=ARCH_NAMES,
                        help='LRL architecture: ')

    parser.add_argument('--arch',  metavar='ARCH', default='MC_UNet',
                        choices=ARCH_NAMES,
                        help='MC_UNet architecture: ')
    parser.add_argument('--noise_branch', action='store_true')

    # dataset 
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--input_crop', default=128, type=int,
                        help='image width')
    parser.add_argument('--depth', default=32, type=int,
                        help='image depth')
    parser.add_argument('--M', default=1, type=int,
                        help='number of sampling')
    
    parser.add_argument('--dataset', default='AtriaSeg',
                        help='dataset name')


    parser.add_argument('--train_txt', default='./train_AtriaSeg.txt',
                        help='text file showing the patient id used for training')

    parser.add_argument('--val_txt', default='./val_AtriaSeg.txt',
                        help='text file showing the patient id used for validation')

    
    parser.add_argument('--img_ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='png',
                        help='mask file extension')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss:  (default: BCEDiceLoss)')
      
    # optimizer
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate') 
    parser.add_argument('--label_factor_semi', default=0.1, type=float,
                        help='percentaget of labeld volume')

    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay') 

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-7, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='50,80', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--pseduo_threshold',default=0.9,type=float)
    parser.add_argument('--ramp_epoch_rate',default=0.5,type=float)
    parser.add_argument('--with_dice',default=1,type=int)
    parser.add_argument('--training_labeled_txt', type=str)
    
    config = parser.parse_args()

    return config


def data_collate(batch):
    input=None
    target = None
    input_paths = None
    total_num =0
    num_per_patient = []
    for info in batch:
      if total_num==0:
        input = torch.from_numpy(info[0]).unsqueeze(0)
        target = torch.from_numpy(info[1]).unsqueeze(0)
        input_paths = info[3]
      else:
        input = torch.cat((input, torch.from_numpy(info[0]).unsqueeze(0)))
        target = torch.cat((target, torch.from_numpy(info[1]).unsqueeze(0)))
        input_paths = np.dstack((input_paths, info[3]))
      num_per_patient.append(info[2])
      total_num+=1

    return input.float(), target,  num_per_patient, input_paths, info[4]


def train(config, train_loader, model, model_seg, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    if epoch > int(config['epochs']/2):
          model_seg.train()
          model.eval()
    else:
          model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, num_per_p, paths, patient in train_loader:#every batch
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)   

      with torch.no_grad():
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        

      KLD = 0.0
      crit = 0.0
      crit_dice = 0.0
      crit_seg = 0.0
      crit_seg_dice = 0.0
      crit_unlabel = 0.0
      crit_dice_unlabel = 0.0
      batch_unlabel = 0
      iou = 0.0 
      recon = 0.0 
      batch_ = 0.0

      input_ = torch.transpose(input_var, 1, 2)
      target_ = torch.transpose(target_var, 1, 2)

      path = paths[0][0] 

      labeled_patient = []
      for ele in patient:
        labeled_patient.append(ele[0].split('/')[-2])

      if epoch > int(config['epochs']/2):
          # train lrl
          with torch.no_grad():
            output,_,_,_ = model(input_, M = config['M'])
            if config['M'] > 1:
              output = output.view(config['M'], input_.size()[-5], config['num_classes'], input_.size()[-3], input_.size()[-2], input_.size()[-1])
              output_pseudo = torch.softmax(output, dim=2).mean(0).detach()
            else:
              output_pseudo = torch.softmax(output, dim=1).detach()

          out_seg = model_seg(input_)

          target_pseudo = torch.empty(0).cuda()
          target_real = torch.empty(0).cuda()
          output_labeled = torch.empty(0).cuda()
          output_unlabeled = torch.empty(0).cuda()
          batch_ = out_seg.size()[0]#batch_size


          name_p = path[0].split('/')[-3]
          for i in range(output.size(0)): 
            try:
              name_p = path[i*config['depth']].split('/')[-3]
            except:
              continue
            # if the case is not in the labeled data, we use pseudo labels. Otherwise, we use pseudo labels.

            if name_p not in labeled_patient:
                target_pseudo = torch.cat([target_pseudo, output_pseudo[i:i+1, :,:,:,:]])
                output_unlabeled = torch.cat([output_unlabeled, out_seg[i:i+1, :, :, :, :]])
            else:
                target_real = torch.cat([target_real, target_[i:i+1, :,:,:,:]])
                output_labeled = torch.cat([output_labeled, out_seg[i:i+1, :, :, :, :]])    

          if min(target_pseudo.shape) != 0:
              crit_seg += criterion[0](output_unlabeled, target_pseudo, num_classes=config['num_classes'])
              crit_seg_dice += criterion[1](output_unlabeled, target_pseudo)
              iou += dice_coef(torch.softmax(output_unlabeled, dim=1), target_pseudo)
          if min(target_real.shape) != 0:
              crit_seg += criterion[0](output_labeled, target_real, num_classes=config['num_classes'])
              crit_seg_dice += criterion[1](output_labeled, target_real)
              iou += dice_coef(torch.softmax(output_labeled, dim=1), target_real)
      else:
            consistency_weight = sigmoid_rampup(epoch,int(config['epochs']/2*config['ramp_epoch_rate']))
            if config['noise_branch']:
              out1, out_noise, mean, covar, x_ori = model(input_, noise_branch = True)

              out_noise_soft = F.softmax(out_noise, dim=1)
              out_soft = F.softmax(out1, dim=1)

              loss_regularization = F.mse_loss(out_soft, out_noise_soft)
            else:  
              out1, mean, covar, x_ori = model(input_)
            
            output = out1
            
            # reconstruction loss & KLD loss
            recon = F.mse_loss(x_ori, input_, reduction='sum')/(x_ori.size(-1)*x_ori.size(-2)*x_ori.size(-3))

            B, depth, D, _ = covar.size()
            mean_view = mean.view(-1, D)
            covar_view = covar.view(-1, D, D)

            prec_matrix_view = torch.linalg.inv(covar_view)
            prec_matrix = prec_matrix_view.view(B, -1, D, D) 

            term1 = torch.logdet(torch.linalg.inv(prec_matrix.sum(1)))  
            term2 = torch.linalg.inv(prec_matrix.sum(1)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            term3_2 =  torch.linalg.inv(torch.bmm(prec_matrix.sum(1), prec_matrix.sum(1)))  
            term3_1 = torch.bmm(prec_matrix.view(-1, D, D), mean_view.unsqueeze(2)).view(-1, depth, D).sum(1)
            term3 = torch.bmm(torch.bmm(term3_1.unsqueeze(1), term3_2), term3_1.unsqueeze(2)).squeeze(2).squeeze(1)
  
            KLD += -0.5 * torch.sum(D + term1.sum() - term2.sum() - term3.sum())

            # ce & dice loss
            for i in range(output.size(0)):
              try:
                name_p = path[i*config['depth']].split('/')[-3]
              except:
                continue

              if name_p not in labeled_patient:
                output_pred = torch.softmax(output[i:i+1,:,:,:,:], dim = 1).detach()

                crit_unlabel += criterion[0](output[i:i+1,:,:,:,:], output_pred, num_classes=config['num_classes'])
                if config['with_dice']:
                  crit_dice_unlabel += criterion[1](output[i:i+1,:,:,:,:], output_pred)
                batch_unlabel += 1

              else:
                crit += criterion[0](output[i:i+1,:,:,:,:], target_[i:i+1,:,:,:,:], num_classes=config['num_classes'])
                crit_dice += criterion[1](output[i:i+1,:,:,:,:], target_[i:i+1,:,:,:,:])
                iou += dice_coef(torch.softmax(output[i:i+1,:,:,:,:], dim=1), target_[i:i+1,:,:,:,:])
                batch_ += 1
        
      # total loss
      if epoch <= int(config['epochs']/2):
        loss = 0.005 * KLD/input_.size(0) + recon/input_.size(0)
        if batch_ > 0:
          loss += crit/batch_ + 2 * crit_dice/batch_
        if batch_unlabel > 0:
          loss += consistency_weight * (crit_unlabel/batch_unlabel + 2 * crit_dice_unlabel/batch_unlabel)
        if config['noise_branch']:
          loss += consistency_weight * loss_regularization
      else:
          loss =  crit_seg/input_var.size(0)  + 2 * crit_seg_dice/len(num_per_p)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      avg_meters['loss'].update(loss.item())
      if batch_ > 0:
          avg_meters['iou'].update(iou/batch_)

      postfix = OrderedDict([
          ('loss', avg_meters['loss'].avg),
          ('iou', avg_meters['iou'].avg),
      ])
      pbar.set_postfix(postfix)
      pbar.update(1)

    pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, model_seg, criterion, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
 
    model.eval()
    model_seg.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        if epoch <= int(config['epochs']/2):
          for input, target, _, _, _, in val_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            input_ = torch.transpose(input_var, 1, 2)
            target_ = torch.transpose(target_var, 1, 2)
        
            output = model(input_, seg_only=True)

            loss = criterion[0](output, target_, num_classes=config['num_classes'])/output.size(0) + criterion[1](output, target_)
            iou = dice_coef(torch.softmax(output, dim=1), target_)
 
            avg_meters['loss'].update(loss.item())
            avg_meters['iou'].update(iou)
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        else:
          for input, target,  _, _, _, in val_loader:
                T = 5
                out_seg = None
                out_seg_ = None
           
                for ii in range(T):
                   input = input.cuda(non_blocking=True)
                   target = target.cuda(non_blocking=True)
                 

                   input_var = torch.autograd.Variable(input)
                   target_var = torch.autograd.Variable(target)
                 
                   input_ = torch.transpose(input_var, 1, 2)
                   target_ = torch.transpose(target_var, 1, 2)
                  
                   out_map = model_seg(input_)
                   score_map = torch.softmax(out_map, dim=1) 
                   if ii == 0:
                     out_seg_ = out_map
                     out_seg = score_map
                   else:
                     out_seg_ = out_seg_ + out_map
                     out_seg = out_seg + score_map

                output = out_seg_/T
                loss = criterion[0](output, target_, num_classes=config['num_classes']) +  criterion[1](output, target_)
                output = out_seg/T
                iou = dice_coef(output, target_)

                avg_meters['loss'].update(loss.item())
                avg_meters['iou'].update(iou)

                postfix = OrderedDict([
                 ('loss', avg_meters['loss'].avg),
                 ('iou', avg_meters['iou'].avg),
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    ### DDP setting
    dist.init_process_group(backend="nccl")

    print('CUDA Device count: ', torch.cuda.device_count())

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print('I am rank %d in this world of size %d!' % (local_rank, world_size))


    ### config
    if config['name'] is None:
      config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('%s%s' % (config['model_save_dir'], config['name']), exist_ok=True)

    # print('-' * 20)
    # for key in config:
    #     print('%s: %s' % (key, config[key]))
    # print('-' * 20)

    with open('%s/%s/config.yml' % (config['model_save_dir'], config['name']), 'w') as f:
        yaml.dump(config, f)

    ### create model
    if local_rank==0:
      print("=> creating model %s" % config['archG'])
    model = archs.__dict__[config['archG']](config['num_classes'],
                                           config['input_channels'],
                                           config['input_crop'],
                                           config['input_crop'])

    if local_rank==0:
      print("=> creating model %s" % config['arch'])
    model_seg = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],)
    
  

    model_seg = model_seg.cuda()
    model_seg = SyncBatchNorm.convert_sync_batchnorm(model_seg)
    model_seg = DistributedDataParallel(model_seg, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = model.cuda()
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
       
    ### criterion, optimizer and scheduler
    criterion_bce =  losses.BceLossThres(threshold = config['pseduo_threshold']).cuda()
    criterion_dice = losses.DiceLoss().cuda()

    criterion = [criterion_bce, criterion_dice]
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    params_seg = filter(lambda p: p.requires_grad, model_seg.parameters())

    optimizer = optim.Adam(
            params, lr=config['lr'] * 1.0, weight_decay=config['weight_decay'])

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    ### Data loading code

    train_transform = transforms.Compose([  
           transforms.Resize(256),
           transforms.CenterCrop(160),
           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        ])

    val_transform = transforms.Compose([  
            transforms.Resize(256),
            transforms.CenterCrop(160),
        ])

    train_dataset = Dataset(
        data_txt = config['train_txt'],
        img_ext = config['img_ext'],
        mask_ext=config['mask_ext'],
        semi_setting=True, 
        training_labeled_txt=config['training_labeled_txt'],
        label_factor_semi=config['label_factor_semi'],
        transform=train_transform,
        rotate_flip=True,
        random_whd_crop =True,
        crop_hw=config['input_crop'],
        depth=config['depth'],
        num_classes = config['num_classes']) 


    val_dataset = Dataset(
        data_txt = config['val_txt'],
        img_ext = config['img_ext'],
        mask_ext=config['mask_ext'],
        semi_setting=False, 
        label_factor_semi = None,  
        transform=val_transform,
        rotate_flip=False,
        random_whd_crop = True,
        crop_hw = config['input_crop'],
        depth = config['depth'],
        num_classes = config['num_classes'])

    train_sampler = DistributedSampler(train_dataset)
     
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn = data_collate,
        num_workers=config['num_workers'],
        drop_last=True)
     
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn = data_collate,
        num_workers=config['num_workers'],
        drop_last=False)
 
 
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0

    ### train for one epoch
    trigger = 0
    # save labeled and unlabeled patients into txt
    patient_label = train_dataset.ret_patient_label()
    patient_unlabel = train_dataset.ret_patient_unlabel()
    with open('%s/%s/labeled_patient.txt' % (config['model_save_dir'], config['name']), 'w') as f:
      for line in patient_label:
        print(line[0], file = f, end = ' ')
        print(line[1], file = f)
    with open('%s/%s/unlabeled_patient.txt' % (config['model_save_dir'], config['name']), 'w') as f:
      for line in patient_unlabel:
        print(line[0], file = f, end = ' ')
        print(line[1], file = f) 

    for epoch in range(config['start_epoch'], config['epochs']):
        train_sampler.set_epoch(epoch)
        if local_rank==0:
          print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # beginning of stage two
        if epoch == int(config['epochs']/2 + 1):
           # resume LRL for MC_UNet training if needed
           if config['start_epoch'] == epoch:    
              model_save_path = '%s/%s/model_121.pth' % (config['model_save_dir'], config['name'])
              model.load_state_dict(torch.load(model_save_path)['state_dict'])
              if local_rank==0:
                print('loading model from', model_save_path)

           best_iou = 0

           # initial optimizer and scheduler for MC_UNet
           optimizer = optim.Adam(params_seg, lr=config['lr'] * 5, weight_decay=config['weight_decay'] * 0.1)

           if config['scheduler'] == 'CosineAnnealingLR':
             scheduler = lr_scheduler.CosineAnnealingLR(
               optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
           elif config['scheduler'] == 'ReduceLROnPlateau':
             scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
           elif config['scheduler'] == 'MultiStepLR':
             scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
           elif config['scheduler'] == 'ConstantLR':
             scheduler = None
           else:
             raise NotImplementedError

        # first train LRL, then train MC_UNet
        train_log = train(config, train_loader, model, model_seg, criterion, optimizer, epoch)
        val_log = validate(config, val_loader, model, model_seg, criterion, epoch)
     
        if config['scheduler'] == 'CosineAnnealingLR' or config['scheduler'] == 'MultiStepLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
      
        if local_rank==0:
          print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        # logging
        if local_rank == 0:
          log['epoch'].append(epoch)
          log['lr'].append(f"{optimizer.state_dict()['param_groups'][0]['lr']:.2e}")
          log['loss'].append(train_log['loss'])
          log['iou'].append(train_log['iou'])
          log['val_loss'].append(val_log['loss'])
          log['val_iou'].append(val_log['iou'])

          pd.DataFrame(log).to_csv('%s/%s/log.csv' %
                                  (config['model_save_dir'], config['name']), index=False)

        trigger += 1

        # saving model
        if local_rank == 0:
          # last lrl model
          if epoch == int(config['epochs']/2):
              save_checkpoint({
                  'epoch': epoch + 1,
                  'arch': config['archG'],
                  'state_dict': model.state_dict(),
                  'best_iou': best_iou,
                  'optimizer' : optimizer.state_dict(),
              }, filename='%s/%s/model_%d.pth' % (config['model_save_dir'], config['name'],epoch+1))

          # last two MC_UNet model
          if epoch >= config['epochs'] - 2:
              best_iou = val_log['iou']
          
              save_checkpoint({
                  'epoch': epoch + 1,
                  'arch': config['arch'],
                  'state_dict': model_seg.state_dict(),
                  'best_iou': best_iou,
                  'optimizer' : optimizer.state_dict(),
              }, filename='%s/%s/model_seg_%d.pth' % (config['model_save_dir'], config['name'],epoch+1))

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

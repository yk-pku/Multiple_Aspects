import argparse
import os
from glob import glob
 
import torch
import torch.backends.cudnn as cudnn
import yaml
import numpy as np 
import torchvision.transforms as transforms 
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import dice_coef, Jaccord, HD, ASD, dice_coef_per_slice
#from utils import AverageMeter
from collections import OrderedDict
from utils import test_single_case, AverageMeter, save_seg_img

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', default='/raid/D/bayes/bayes_AtriaSeg_8/',
                        help='model directory')
    parser.add_argument('--input_crop', default=128, type=int,
                        help='image width')
    parser.add_argument('--depth', default=32, type=int,
                        help='image depth')
    parser.add_argument('--test_txt', default='./val_AtriaSeg.txt',
                        help='text file showing the patient id used for validation')
    parser.add_argument('--gpu_id', default=0, type=int,
                        metavar='N', help='setting gpu id') 
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_epoch', default='', type=str,
                        help='number of epoch')    
    parser.add_argument('--skip_metric',action='store_true',
                        help = 'save_image_or_not')
    parser.add_argument('--save_image',action='store_true',
                        help = 'save_image_or_not')
    parser.add_argument('--save_image_interval',default = 1, type = int,
                        help = 'save_image_interval')                
   
    args = parser.parse_args()

    return args
 

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

    return input.float(), target, num_per_patient, input_paths, info[4]




def main():
    args = parse_args()
    model_dir = args.model_dir

    yml = os.path.join(model_dir, 'config.yml')

    with open(yml, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print('-'*20)
    # for key in config.keys():
    #     print('%s: %s' % (key, str(config[key])))
    # print('-'*20)

    cudnn.benchmark = True
 
    print("=> creating model %s" % config['arch'])
    model_seg = archs.__dict__[config['arch']](args.num_classes,
                                           config['input_channels'])
        
 
    model_seg = model_seg.cuda()
    model_name = 'model_seg%s.pth'%args.num_epoch
  
    model_seg_path = os.path.join(model_dir, model_name)

    checkpoint = torch.load(model_seg_path) 
    pretrain_dict = checkpoint['state_dict']
    
    new_dict = OrderedDict()

    for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
                new_dict[k] = v
 
    model_dict = model_seg.state_dict()
    model_dict.update(new_dict)
    model_seg.load_state_dict(model_dict)


    model_seg.eval()

    torch.cuda.set_device(args.gpu_id)

    test_transform = transforms.Compose([  
           transforms.Resize(256),
            transforms.CenterCrop(160),
        ])

    test_dataset = Dataset(
        data_txt = args.test_txt,
        img_ext = 'png',
        mask_ext= 'png',
        semi_setting=False, 
        label_factor_semi = None,  
        transform=test_transform,
        rotate_flip=False,
        random_whd_crop = False,
        crop_hw = 128,
        depth = None)


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn = data_collate,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meters = {'dice': AverageMeter(),
                  'jaccord': AverageMeter(), 
                   'hd95': AverageMeter(),
                    'asd': AverageMeter()}

    with torch.no_grad():
        img_cnt = -1
        model_img_dir = ''
        red = [0,0,255]

        for input, target, _, paths, _, in tqdm(test_loader, total=len(test_loader)):
            #save image setting
            img_cnt += 1
            path = paths[0][0]
            name_p = path[0].split('/')[-3]
            if img_cnt % args.save_image_interval == 0 and args.save_image:
                # model_gt_once = 'gt_results/'
                # os.makedirs(os.path.join(model_gt_once,name_p))
                # save_seg_img(input, target, os.path.join(model_gt_once,name_p))
                # continue
                model_img_dir = model_dir + '/' + args.test_txt.split('/')[-1][:-4] + '/' + name_p
                if not os.path.exists(model_img_dir):
                    os.makedirs(model_img_dir)

            input = input.cuda()
            target = target.cuda()
            T = 1
            out_seg = None
            out_seg_ = None
           
            for ii in range(T):
             input = input.cuda(non_blocking=True)
             target = target.cuda(non_blocking=True)
             input_var = torch.autograd.Variable(input)
             target_var = torch.autograd.Variable(target)
             input_ = torch.transpose(input_var, 1, 2)
             target_ = torch.transpose(target_var, 1, 2)

             out_map, score_map = test_single_case(model_seg, input_, 8,  8, patch_size=(args.input_crop, args.input_crop, args.depth), num_classes=args.num_classes) 
             if ii == 0:
               out_seg = score_map
               out_seg_ = out_map
             else:
               out_seg = out_seg + score_map
               out_seg_ = out_seg_ + out_map

            output = out_seg/T


            if args.save_image and img_cnt % args.save_image_interval == 0:
                dice_slice = dice_coef_per_slice(output, target_)
                save_seg_img(input, output, model_img_dir , dice_slice)
                
            if not args.skip_metric:
                dice = dice_coef(output, target_)
                jaccord = Jaccord(output, target_)
                hd = HD(output, target_)
                asd = ASD(output, target_)

                avg_meters['dice'].update(dice, input.size(0))
                avg_meters['jaccord'].update(jaccord, input.size(0))
                avg_meters['hd95'].update(hd, input.size(0))
                avg_meters['asd'].update(asd, input.size(0))
   

        if not args.skip_metric:
            with open('%stest_result.txt' % model_dir,'a') as f:
                print(model_name,file=f)
                print('Dice: %.4f' % avg_meters['dice'].avg, file=f)
                print('Jaccord: %.4f' % avg_meters['jaccord'].avg, file=f)
                print('hd95: %.4f' % avg_meters['hd95'].avg, file=f)
                print('asd: %.4f' % avg_meters['asd'].avg, file=f)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

import os
  
import numpy as np
import torch
import torch.utils.data
import cv2
import random 
from PIL import Image 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_txt, img_ext, mask_ext, transform=None,  semi_setting=True, training_labeled_txt = None, label_factor_semi=0.2, rotate_flip=True, depth=96, crop_hw = 96, random_whd_crop=True,  num_classes=2):

        self.data_txt = data_txt
        self.transform = transform
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.patient = [] 
        self.random_whd_crop = random_whd_crop
        self.depth = depth
        self.crop_hw = crop_hw
        self.semi_setting = semi_setting
        self.patient_label =[]
        self.patient_unlabel = [] 
        self.rotate_flip=rotate_flip
        self.num_classes = num_classes
        self.training_labeled_txt = training_labeled_txt

        w = open(self.data_txt)
        # store images and label maps
        for ele in w.readlines():
          id = ele.split(' ')[0]
          id_l = ele.split(' ')[1].split('\n')[0]
          self.patient.append([id, id_l])
 
        num = len(self.patient)
        random.shuffle(self.patient)

        if self.semi_setting:
          # from used training_labeled_txt
          if self.training_labeled_txt is not None:
            with open(self.training_labeled_txt, 'r') as f:
              for ele in f.readlines():
                id = ele.split(' ')[0]
                id_l = ele.split(' ')[1].split('\n')[0]
                self.patient_label.append([id, id_l])
              for cur_patient in self.patient:
                if cur_patient not in self.patient_label:
                  self.patient_unlabel.append(cur_patient)
          # randomly sample
          else:
            num_patient =  int(num * label_factor_semi)
            for i in range(num):
              if i < num_patient:
                self.patient_label.append(self.patient[i])
              else:
                self.patient_unlabel.append(self.patient[i])
          num_ratio = int(len(self.patient_unlabel)/len(self.patient_label))
          # copy to make they have the same length for sampling
          # balanced labeled and unlabeled data
          patient_label_repeat = self.patient_label * (num_ratio  + 1)
          self.patient =  patient_label_repeat + self.patient_unlabel
 
         
    def __len__(self):
        return len(self.patient)
    
    def rotate(self, image, label, angle, center=None, scale=1.0):
   
       (h, w) = image.shape[:2]

       if center is None:
          center = (w / 2, h / 2)

       M = cv2.getRotationMatrix2D(center, angle, scale)
       rotated_image = cv2.warpAffine(image, M, (w, h))
       rotated_label = cv2.warpAffine(label, M, (w, h))
  
       return rotated_image, rotated_label
    
    


    def random_crop(self, image, mask, crop_sz):
      img_sz = image.shape[0]
      random_arr = np.random.randint(img_sz-crop_sz+1, size=2)
      y = random_arr[1]
      x = random_arr[0]
      h = crop_sz 
      image_crop = image[y:y+h, x:x+h, :]
      mask_crop = mask[y:y+h, x:x+h, :]

      return image_crop, mask_crop
    
    def transform_crop(self, image, label, minx, maxx, miny, maxy, output_size=[112, 112]):
 
        w, h = label.shape
 
        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2

        minx = max(minx - 10 - px, 0)
        maxx = min(maxx + 10 + px, w)
        miny = max(miny - 10 - py, 0)
        maxy = min(maxy + 10 + py, h)
  
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy, :]
        label = label[minx:maxx, miny:maxy]

        label = np.expand_dims(label, -1)

        return image, label

    def ret_patient_label(self):
        return self.patient_label
    
    def ret_patient_unlabel(self):
        return self.patient_unlabel
    
    def __getitem__(self, idx):
        patient = self.patient[idx]
        imgs = []
        masks = []
        img_list = os.listdir(patient[0]) 

        num = len(img_list)
        base_name = patient[0].split('/')[-1]
        base_name_l = patient[1].split('/')[-1].split('\n')[0]

        nums = []

        image_shape = None
        mask_shape = None

        for ele in img_list:
           ele1 = ele.split('-')[-1]
           num_ = int(ele1.split('.')[0])
           nums.append(num_)

        base_num = min(nums)
        img_channel = 3 
        img_paths = []
        num_collect = 0
 
        for i in range(0, num):#一张张2D切片读
           ele = base_name + '-' + str(base_num+i)+'.'+self.img_ext
           ele_l = base_name_l + '-' + str(base_num+i)+'.'+self.mask_ext
           img = os.path.join(patient[0], ele)
           img_ = Image.open(img).convert('RGB') 

           anno = os.path.join(patient[1], ele_l)
           mask_ = Image.open(anno).convert('L')
            
           img_sum = np.sum(np.array(img_).astype('float32'))
           mask_sum = np.sum(np.array(mask_).astype('float32'))
 
           if img_sum == 0.0 or mask_sum < 1.0:#过滤无物体的slice
                continue
 
           imgs.append(img_)
           masks.append(mask_)
           img_paths.append(img)
           num_collect += 1
 
        for i in range(len(imgs)): #每张slice做transform
            img_ = imgs[i]
            mask_ = masks[i]
           
            if self.transform is not None:
              imgs[i] = np.array(self.transform(img_))
              mask_ = np.expand_dims(np.array(self.transform(mask_)), -1)
 

            if np.max(mask_.astype('float32')) != 0.0:#label
               masks[i] = mask_.astype('float32') / np.max(mask_.astype('float32')) * (self.num_classes - 1)
            else:
               masks[i] = mask_

         # img 归一化
        img_mean = np.mean(imgs)
        img_std = np.std(imgs)

        for i in range(len(imgs)):
            if np.max(imgs[i].astype('float32')) > 0.0: 
              imgs[i] = (imgs[i].astype('float32') - img_mean) / img_std
 
        image_shape = imgs[0].shape
        mask_shape  = masks[0].shape
      
         # 随机裁剪
        if self.random_whd_crop:
         n_frame = len(imgs)
         if n_frame >= int(self.depth):
           start = random.randint(0, n_frame-self.depth)
           imgs = imgs[start:start+self.depth]
           masks = masks[start:start+self.depth]
           img_paths= img_paths[start:start+self.depth]
         else:
           N = self.depth - n_frame
           zero_img = np.zeros(image_shape)
           zero_mask = np.zeros(mask_shape)
           for i in range(N):
               imgs.append(zero_img)
               masks.append(zero_mask)
               img_paths.append(' ')

        img = np.array(np.dstack(imgs))
        mask = np.array(np.dstack(masks))
        img_paths = np.dstack(img_paths)

        try:
         if self.random_whd_crop: 
            img, mask = self.random_crop(img, mask, self.crop_hw) 


         if self.rotate_flip:
          a = np.random.randint(2)
          if a == 0:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
          
          b  = np.random.randint(2)
          if b == 0:
            angle = np.random.uniform(-10, 10)
            img, mask = self.rotate(img, mask, angle)
           
        finally:
         img = img.transpose(2, 0, 1)
         mask = mask.transpose(2, 0, 1)

         img = np.expand_dims(img, 1)
         img = np.reshape(img, (-1, img_channel, img.shape[-2], img.shape[-1]))
           
 
         mask = np.expand_dims(mask, 1)
         mask = np.reshape(mask, (-1, 1, mask.shape[-2], mask.shape[-1]))
 
         mask_ones = np.ones(mask.shape)

 
         mask_one_hot = None
 
         for i in range(self.num_classes): 
             if mask_one_hot is None:
                mask_one_hot =  (mask == ((i) * mask_ones)).astype('int')
             else:
                mask_one_hot = np.concatenate((mask_one_hot, (mask == (i * mask_ones)).astype('int')), axis=1)
 
        return img, mask_one_hot, num_collect, img_paths, self.patient_label


if __name__ == '__main__':
    import torchvision.transforms as transforms 
    train_transform = transforms.Compose([  
           transforms.Resize(256),
           transforms.CenterCrop(160),
           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        ])
   
    train_dataset =  Dataset(data_txt = './dataset_split/train_AtriaSeg.txt',
                            img_ext='png',
                            mask_ext='png',
                            semi_setting=True, 
                            training_labeled_txt=None,
                            label_factor_semi=0.1,
                            transform=train_transform,
                            crop_hw=128,
                            depth=32,
                            num_classes=2)
   
    labeled_data = train_dataset.ret_patient_label()
   
    for data in train_dataset:
      import pdb; pdb.set_trace()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True) 
   
    for zip_data in train_loader:
      import pdb; pdb.set_trace()
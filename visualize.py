import cv2
import numpy as np
import torchvision.transforms as transforms 
import os

patients = [  "6799D6LEBH3NSRV1KH27",
  "KSNYHUBHHUJTYJ14UQZR",
  "38CWS74285MFGZZXR09Z",
  "5UB5KFD2PK38Z4LS6W80",
  "0RZDK210BSMWAA6467LU",
  "VDOF02M8ZHEAADFMS6NP",
  "DYXSCIWHLSUOZIDDSZ40",
  "WNPKE0W404QE9AELX1LR",
  "GSC9KNY0VEZXFSGWNF25",
  "CLXFYOBQDCVXQ9P7YC07",
  "MVKIPGBKTNSENNP1S4HB",
  "8M99G0JLAXG9GLPV0O8G",
  "9DCM2IB45SK6YKQNYUQY",
  "V0MZOWJ6MU3RMRCV9EXR",
  "5FKQL4K14KCB72Y8YMC2",
  "IQYKPTWXVV9H0IHB8YXC",
  "CBIJFVZ5L9BS0LKWE8YL",
  "1D7CUD1955YZPGK8XHJX",
  "YDKD1HVHSME6NVMA8I39",
  "1GU15S0GJ6PFNARO469W"]

def apply_transform_opencv(image):
    # 调整大小
    resized_image = cv2.resize(image, (256, 256))

    # 中心裁剪
    h, w = resized_image.shape[:2]
    start_h = (h - 160) // 2
    start_w = (w - 160) // 2
    cropped_image = resized_image[start_h:start_h+160, start_w:start_w+160]
    return cropped_image

def overlay_segmentation(rgb_image_path, binary_mask_path, output_path):
    # 读取RGB图像
    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = apply_transform_opencv(rgb_image)

    # 读取二进制分割图
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    binary_mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    # 将二进制分割图转为3通道红色图，以便与RGB图像叠加
    binary_mask_colored  = np.zeros_like(binary_mask_colored)
    binary_mask_colored [binary_mask == 255] = [0, 0, 255]

    #import pdb; pdb.set_trace()
    # 将分割结果叠加在原始RGB图像上
    overlaid_image = cv2.addWeighted(rgb_image, 0.5, binary_mask_colored, 0.5, 0)

    # 保存结果
    cv2.imwrite(output_path, overlaid_image)


if __name__ == "__main__":

    rgb_root = '/workspace/GBDL/AtrialSeg'

    mask_root = '/workspace/GBDL/results_10%/test_new_train/'

    for idx, patient in enumerate(patients):
        img_patient_path = os.path.join(rgb_root,patient,'lgemri.nrrd')
        img_slices = os.listdir(img_patient_path)
        img_slices = sorted(img_slices, key = lambda x: int(x.split('-')[1].strip('.png')))

        mask_patient_path = os.path.join(mask_root,'test_AtriaSeg',patient)
        mask_slices = os.listdir(mask_patient_path)
        mask_slices = sorted(mask_slices, key=lambda x: int(x.split('_')[1]))

        out_patient_path = os.path.join(mask_root,'test_AtriaSeg_overlaid_1',patient)
        if not os.path.exists(out_patient_path):
            os.makedirs(out_patient_path)

        for img_slice, mask_slice in zip(img_slices,mask_slices):
            # 输入RGB图像路径
            rgb_image_path = os.path.join(img_patient_path,img_slice)

            # 输入二进制分割图路径
            binary_mask_path = os.path.join(mask_patient_path,mask_slice)

            # 输出叠加结果路径
            output_path = os.path.join(out_patient_path,mask_slice)

            # 调用函数进行叠加
            overlay_segmentation(rgb_image_path, binary_mask_path, output_path)
        if idx == 0:
            break

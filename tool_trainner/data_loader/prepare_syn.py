import argparse
import glob
import h5py
import numpy as np
import cv2
import os
import random

def show_img_cv2_plt(image) :
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def make_dataset(args):
    if not args.test : 
        raw_folder = os.path.join(args.output_path+'/train/raw/')
        rgb_folder = os.path.join(args.output_path+'/train/rgb/')
    else:
        raw_folder = os.path.join(args.output_path+'/test/raw/')
        rgb_folder = os.path.join(args.output_path+'/test/rgb/')

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    
    if not args.test : 
        sections = ['DIV2K/DIV2K_train_HR/','KITTI_detection/data_object_image_2/training/image_2/',\
                    'MS_COCO/train2017'] #,'/KITTI_depth/KITTI_sc/2011_10_03_drive_0034_sync_02/']
    else:
        sections = ['DIV2K/DIV2K_valid_HR/','/KITTI_detection/data_object_image_2/testing/image_2/', \
                        'MS_COCO/val2017'] #,'/KITTI_depth/KITTI_sc/2011_09_30_drive_0034_sync_02/']

    idx = 0
    for section in sections:        
        img_dir_rgb       = args.images_dir_rgb+'/'+section+'/'
        img_dir_raw       = args.images_dir_raw+'/'+section+'/'    
        img_path_rgb_list = sorted(glob.glob('{}/*.png'.format(img_dir_rgb)))
        img_path_raw_list = sorted(glob.glob('{}/*.png'.format(img_dir_raw)))

        if(img_path_rgb_list == []) : 
            img_path_rgb_list = sorted(glob.glob('{}/*.jpg'.format(img_dir_rgb)))
        if not args.test :
            img_path_rgb_list  = img_path_rgb_list[:5000]
            img_path_raw_list  = img_path_raw_list[:5000]
        else: 
            img_path_rgb_list  = img_path_rgb_list[:400]
            img_path_raw_list  = img_path_raw_list[:400]
        if(img_path_raw_list == []) : 
            print('wrong directory')

        # if len(img_path_rgb_list) > 1000 : 
        #     img_path_rgb_list      = img_path_rgb_list[:1000]
        #     img_path_raw_list  = img_path_raw_list[:1000]

        for image_rgb_path, image_raw_path in zip(img_path_rgb_list, img_path_raw_list):
            if (idx%100 == 0) : 
                print('processed :{0} th file : {1}'.format(idx, image_rgb_path))

            # 1. read RGB/Bayer images
            img_rgb_ = cv2.imread(image_rgb_path, cv2.IMREAD_UNCHANGED) # BGR
            img_rgb_ = cv2.cvtColor(img_rgb_, cv2.COLOR_BGR2RGB)
            img_raw_  = cv2.imread(image_raw_path, cv2.IMREAD_UNCHANGED)
            if img_raw_.dtype == 'uint16' :
                img_raw_  = (img_raw_/1023*255).clip(0,255).astype(np.uint8)

            # 3ch R(G1+G2)B Bayer --> 4ch RG1G2B Bayer
            H_,W_,_ = img_rgb_.shape
            H_2,W_2,_ = img_raw_.shape 

            if H_ != H_2 : 
                print(image_rgb_path)
                print(image_raw_path)
                continue
            elif W_ != W_2:
                print(image_rgb_path)
                print(image_raw_path)
                continue

            if H_//32*32 != H_ :
                img_rgb_ = img_rgb_[:H_//32*32, :, :]
                img_raw_ = img_raw_[:H_//32*32, :, :]
            if W_//32*32 != W_ :
                img_rgb_ = img_rgb_[:, :W_//32*32, :]
                img_raw_ = img_raw_[:, :W_//32*32, :]

            H_,W_,_ = img_rgb_.shape
            img_rgb = img_rgb_ # HxWx3
            img_raw = np.zeros((H_//2,W_//2,4)) # HxWx4 ==> H/2xW/2x4
            img_raw[:,:,0] = img_raw_[1::2, 0::2, 0] # R
            img_raw[:,:,1] = img_raw_[0::2, 0::2, 1] # G1
            img_raw[:,:,2] = img_raw_[1::2, 1::2, 1] # G2 
            img_raw[:,:,3] = img_raw_[0::2, 1::2, 2] # B

            if 'DIV2K' in section :
                img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1]//4, img_rgb.shape[0]//4), interpolation=cv2.INTER_CUBIC)
                img_raw = cv2.resize(img_raw, (img_raw.shape[1]//4, img_raw.shape[0]//4), interpolation=cv2.INTER_CUBIC)
            else:
                img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
                img_raw = cv2.resize(img_raw, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            # h/2 x w/2 x 4 --> h x w (original bayer) 
            H_,W_,_ = img_rgb.shape
            img_raw_ = np.zeros((H_,W_))
            img_raw_[1::2, 0::2] = img_raw[:, :, 0]
            img_raw_[0::2, 0::2] = img_raw[:, :, 1]
            img_raw_[1::2, 1::2] = img_raw[:, :, 2]
            img_raw_[0::2, 1::2] = img_raw[:, :, 3]

            img_raw_ = img_raw_.clip(0,255).astype(np.uint8)
            img_rgb  = img_rgb.clip(0,255).astype(np.uint8)
            cv2.imwrite(raw_folder+str(idx).zfill(6)+'.png', img_raw_)
            cv2.imwrite(rgb_folder+str(idx).zfill(6)+'.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir-rgb', type=str, default='/SSD1/Dataset_DRLISP/', required=True)
    parser.add_argument('--images-dir-raw', type=str, default='/SSD1/Dataset_DRLISP_RAW/', required=True)
    parser.add_argument('--output-path', type=str, default='./dataset/', required=True)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--augmentation', action='store_true')

    args = parser.parse_args()
    dirpath = os.path.dirname(args.output_path)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    make_dataset(args)
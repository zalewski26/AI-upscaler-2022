import cv2
import os
import patchify
from config import Config

def prepare_data():
    print('Creating training patches...')
    prepare_training_patches(
        in_path=Config.train_path, 
        out_path=Config.train_patch_path, 
        scale_factor=Config.SCALE_FACTOR, 
        interpolation=cv2.INTER_AREA, 
        patch_size=Config.PATCH_SIZE, 
        stride=Config.STRIDE
    )
    print('Done')
    print('Downscaling validation images...')
    prepare_downscaled(
        in_path=Config.valid_path, 
        out_path=Config.valid_down_path, 
        scale_factor=Config.SCALE_FACTOR, 
        interpolation=cv2.INTER_AREA
    )
    print('Done')
    print('Downscaling testing images...')
    prepare_downscaled(
        in_path=Config.test_path, 
        out_path=Config.test_down_path, 
        scale_factor=Config.SCALE_FACTOR, 
        interpolation=cv2.INTER_AREA
    )
    print('Done')
    print('---------------------')

def prepare_training_patches(in_path, out_path, scale_factor, interpolation, patch_size, stride):
    "The function responsible for preparing training data from a given set of images."
    if in_path.endswith('/'): in_path = in_path[0:-1]
    if out_path.endswith('/'): out_path = out_path[0:-1]
    os.makedirs(out_path, exist_ok=True)
    lr_path = out_path + f'/lr{scale_factor}'
    lr_up_path = out_path + f'/lr_up{scale_factor}'
    hr_path = out_path + f'/hr{scale_factor}'
    os.makedirs(lr_path, exist_ok=True)
    os.makedirs(lr_up_path, exist_ok=True)
    os.makedirs(hr_path, exist_ok=True)
    
    for filename in os.listdir(in_path):
        img = cv2.imread(in_path + '/' + filename)
        img_name = filename.split('.')[0]
        patches = patchify.patchify(img, (patch_size, patch_size, 3), step=stride)
        id = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                id += 1
                patch = patches[i, j, 0, :, :, :]
                cv2.imwrite(f"{hr_path}/{img_name}_{id}.png", patch)
                h, w, _ = patch.shape
                low_patch = cv2.resize(patch, (w//scale_factor, h//scale_factor), interpolation=interpolation)
                cv2.imwrite(f"{lr_path}/{img_name}_{id}.png", low_patch)
                low_patch = cv2.resize(low_patch, (w, h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{lr_up_path}/{img_name}_{id}.png", low_patch)

def prepare_downscaled(in_path, out_path, scale_factor, interpolation):
    "The function responsible for preparing downscaled images from a given set."
    if in_path.endswith('/'): in_path = in_path[0:-1]
    if out_path.endswith('/'): out_path = out_path[0:-1]
    os.makedirs(out_path, exist_ok=True)
    hr_path = out_path + f'/hr{scale_factor}'
    lr_path = out_path + f'/lr{scale_factor}'
    lr_up_path = out_path + f'/lr_up{scale_factor}'
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(lr_path, exist_ok=True)
    os.makedirs(lr_up_path, exist_ok=True)
    
    for filename in os.listdir(in_path):
        img = cv2.imread(in_path + '/' + filename)
        img_name = filename.split('.')[0]
        h, w, _ = img.shape
        img = cv2.resize(img, (w - (w % scale_factor), h - (h % scale_factor)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{hr_path}/{img_name}.png", img)

        h, w, _ = img.shape
        low_img = cv2.resize(img, (w//scale_factor, h//scale_factor), interpolation=interpolation)
        cv2.imwrite(f"{lr_path}/{img_name}.png", low_img)
        low_img = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{lr_up_path}/{img_name}.png", low_img)


if __name__ == '__main__':
    prepare_data()
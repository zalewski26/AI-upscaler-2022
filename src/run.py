import os
import cv2
import torch
import datetime
import argparse
import numpy as np
import srcnn, srgan
from PIL import Image
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='description')
parser.add_argument('--arch', type=str, choices=['srgan', 'srcnn'], required=True)
parser.add_argument('--channels', type=int, choices=[1, 3], required=True)
parser.add_argument('--scale-factor', dest='scale_factor', type=int, choices=[2, 4], required=True)
parser.add_argument('--img-path', dest='img_path', type=str, required=True)
parser.add_argument('--weights-path', dest='weights_path', type=str, required=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(parser.parse_args())
    arch = args['arch']
    channels = args['channels']
    scale_factor = args['scale_factor']
    img_path = args['img_path']
    weights_path = args['weights_path']

    ct = datetime.datetime.now()
    result_path = f'../output/Test/{arch.upper()}/{ct.year}.{ct.month}.{ct.day}_{ct.hour}.{ct.minute}.{ct.second}'
    os.makedirs(result_path, exist_ok=True)

    model = None
    if arch == 'srcnn':
        model = srcnn.SuperResolutionCNN(channels).to(device)
    elif arch == 'srgan':
        model = srgan.Generator(scale_factor, channels).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    img = cv2.imread(img_path)
    inter_img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
    filename = os.path.basename(img_path)
    img_name = filename.split('.')[0]
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
    if arch == 'srcnn':
        temp_img = cv2.resize(temp_img, (temp_img.shape[1] * scale_factor, temp_img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
    if channels == 1:
        temp_img_Y = torch.zeros((1, 1, temp_img.shape[0], temp_img.shape[1]))
        temp_img_Y[0, 0, :, :] = torch.tensor(temp_img[:, :, 0]) / 255
        if arch == 'srgan':
            temp_img = cv2.resize(temp_img, (temp_img.shape[1] * scale_factor, temp_img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
        pred = model(temp_img_Y.to(device))
        pred = np.array(pred.detach().cpu().numpy(), dtype=np.float32)
        pred *= 255
        pred[pred[:] > 255] = 255
        pred[pred[:] < 0] = 0
        temp_img[:,:,0] = pred
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_YCR_CB2BGR)
    else:
        temp_img /= 255
        temp_img = np.array(temp_img.transpose([2, 0, 1]))
        temp_img = torch.tensor(np.array([temp_img]), dtype=torch.float)
        temp_img = model(temp_img.to(device))
        temp_img = np.array(temp_img[0].detach().cpu().numpy(), dtype=np.float32)
        temp_img = temp_img.transpose([1, 2, 0])
        temp_img *= 255
        temp_img[temp_img[:] > 255] = 255
        temp_img[temp_img[:] < 0] = 0
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

    transform(img).save(f'{result_path}/{img_name}.png','png')
    transform(inter_img).save(f'{result_path}/{img_name}_bicubic.png','png')
    transform(temp_img).save(f'{result_path}/{img_name}_up.png','png')

    compare_image = Image.new('RGB', (temp_img.shape[1] * 3, temp_img.shape[0]))
    compare_image.paste(transform(img), (0, temp_img.shape[0]-img.shape[0], img.shape[1], temp_img.shape[0]))
    compare_image.paste(transform(inter_img), (temp_img.shape[1] * 1, 0, temp_img.shape[1] * 2, temp_img.shape[0]))
    compare_image.paste(transform(temp_img), (temp_img.shape[1] * 2, 0, temp_img.shape[1] * 3, temp_img.shape[0]))
    compare_image.save(f'{result_path}/{img_name}_compare.pdf','PDF', resolution=100)
    
def transform(img, shape=None):
    shape = img.shape if shape == None else shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return T.Compose([
        T.ToPILImage()
    ]) (img)

if __name__ == '__main__':
    main()
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import torchvision.transforms as T

def psnr(img1, img2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    rmse = math.sqrt(np.mean((img2 - img1) ** 2))
    if rmse == 0:
        return 100
    else:
        result = 20 * math.log10(1.0 / rmse)
        return result

def plot_snr(psnr, result_dir):
    plt.figure(figsize=(10, 7))
    plt.plot(psnr, color='green', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(f'{result_dir}/psnr.png')
    plt.close()

def plot_ssim(ssim, result_dir):
    plt.figure(figsize=(10, 7))
    plt.plot(ssim, color='green', label='validation ssim')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(f'{result_dir}/ssim.png')
    plt.close()

def saveAsCsv(snr, ssim, result_dir):
    snr = np.asarray(snr)
    ssim = np.asarray(ssim)
    pd.DataFrame(snr).to_csv(f'{result_dir}/snr.csv')
    pd.DataFrame(ssim).to_csv(f'{result_dir}/ssim.csv')

def sanity_check(model, low_path, high_path, pdf_path, upscaled, channels):
    "Function that performs upscaling on selected set of images."
    model.eval()
    files = os.listdir(high_path)
    result_image = Image.new('RGB', (900, len(files)*300))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        it = 0
        for filename in files:
            low_res = cv2.imread(low_path + '/' + filename)
            high_res = cv2.imread(high_path + '/' + filename)
            temp_res = None

            if channels == 1:
                temp_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2YCR_CB)
                temp_res_Y = torch.zeros((1, 1, temp_res.shape[0], temp_res.shape[1]))
                temp_res_Y[0, 0, :, :] = torch.tensor(temp_res[:, :, 0]) / 255
                if not upscaled:
                    temp_res = cv2.resize(temp_res, (high_res.shape[1], high_res.shape[0]), interpolation=cv2.INTER_CUBIC)

                pred = model(temp_res_Y.to(device))
                pred = np.array(pred.detach().cpu().numpy(), dtype=np.float32)
                pred *= 255
                pred[pred[:] > 255] = 255
                pred[pred[:] < 0] = 0
                temp_res[:,:,0] = pred
                temp_res = cv2.cvtColor(temp_res, cv2.COLOR_YCR_CB2BGR)

            elif channels == 3:
                temp_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB).astype(float)
                temp_res /= 255
                temp_res = temp_res.transpose([2, 0, 1])
                temp_res = torch.tensor(np.array([temp_res]), dtype=torch.float)
                temp_res = model(temp_res.to(device))
                temp_res = np.array(temp_res[0].detach().cpu().numpy(), dtype=np.float32)
                temp_res = temp_res.transpose([1, 2, 0])
                temp_res *= 255
                temp_res[temp_res[:] > 255] = 255
                temp_res[temp_res[:] < 0] = 0
                temp_res = cv2.cvtColor(temp_res, cv2.COLOR_RGB2BGR)
            
            if not upscaled:
                    low_res = cv2.resize(low_res, (high_res.shape[1], high_res.shape[0]), interpolation=cv2.INTER_CUBIC)

            result_image.paste(transform(high_res), (0, it*300, 300, it*300+300))
            result_image.paste(transform(low_res), (300, it*300, 600, it*300+300))
            result_image.paste(transform(temp_res), (600, it*300, 900, it*300+300))
            it += 1
    result_image.save(pdf_path,'PDF',resolution=100.00,save_all=True)

def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    return T.Compose([
        T.ToPILImage(),
        T.Resize(300),
        T.CenterCrop(300),
    ]) (img)
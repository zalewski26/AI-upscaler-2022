import torch
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

def getDataset(lr_path, hr_path, channels):
    if channels == 1:
        return OneChannelDataset(lr_path, hr_path)
    elif channels == 3:
        return ThreeChannelsDataset(lr_path, hr_path)

class OneChannelDataset(Dataset):
    "Dataset intended for use with training on Y channel (YCbCr color space)."
    def __init__(self, lr_path, hr_path):
        super(OneChannelDataset, self).__init__()
        self.lr_imgs = glob.glob(f"{lr_path}/*")
        self.hr_imgs = glob.glob(f"{hr_path}/*")

    def __getitem__(self, index):
        low_res = cv2.imread(self.lr_imgs[index])
        high_res = cv2.imread(self.hr_imgs[index])
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2YCR_CB)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2YCR_CB)
        low_res_Y = np.zeros((1, low_res.shape[0], low_res.shape[1]), dtype=float)
        high_res_Y = np.zeros((1, high_res.shape[0], high_res.shape[1]), dtype=float)
        low_res_Y[0, :, :] = low_res[:, :, 0].astype(float) / 255
        high_res_Y[0, :, :] = high_res[:, :, 0].astype(float) / 255
        return (torch.tensor(low_res_Y, dtype=torch.float), torch.tensor(high_res_Y, dtype=torch.float))

    def __len__(self):
        return (len(self.lr_imgs))
    
class ThreeChannelsDataset(Dataset):
    "Dataset intended for use with training on RGB channels."
    def __init__(self, lr_path, hr_path):
        super(ThreeChannelsDataset, self).__init__()
        self.lr_imgs = glob.glob(f"{lr_path}/*")
        self.hr_imgs = glob.glob(f"{hr_path}/*")

    def __getitem__(self, index):
        low_res = cv2.imread(self.lr_imgs[index])
        high_res = cv2.imread(self.hr_imgs[index])
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB).astype(float)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB).astype(float)
        low_res /= 255
        high_res /= 255
        low_res = low_res.transpose([2, 0, 1])
        high_res = high_res.transpose([2, 0, 1])
        return (torch.tensor(low_res, dtype=torch.float), torch.tensor(high_res, dtype=torch.float))

    def __len__(self):
        return (len(self.lr_imgs))
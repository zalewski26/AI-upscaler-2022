import os
import torch
import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim
from models import SuperResolutionCNN, Generator, Discriminator, GeneratorCriterion
from datasets import getDataset
from utils import psnr, plot_snr, plot_ssim, sanity_check_dir, saveAsCsv
from config import Config

parser = argparse.ArgumentParser(description='Training script for image upscaling neural network models.')
parser.add_argument('--arch', type=str, choices=['srgan', 'srcnn'], required=True)
parser.add_argument('--channels', type=int, choices=[1, 3], required=True)
def main():
    args = vars(parser.parse_args())
    arch = args['arch']
    channels = args['channels']
    result_path = None

    if arch == 'srcnn':
        srcnn_train_dataset = getDataset(Config.train_patch_path_lr_up, Config.train_patch_path_hr, channels)
        srcnn_train_loader = DataLoader(srcnn_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        srcnn_valid_dataset = getDataset(Config.valid_down_path_lr_up, Config.valid_down_path_hr, 3)
        srcnn_valid_loader = DataLoader(srcnn_valid_dataset, batch_size=1, shuffle=True)
        srcnn_model = SuperResolutionCNN(channels).to(Config.device)
        srcnn_optim = optim.Adam(srcnn_model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.MSELoss()
        ct = datetime.datetime.now()
        result_path = f'../output/Training/SRCNN/{ct.year}.{ct.month}.{ct.day}_{ct.hour}.{ct.minute}.{ct.second}'
        os.makedirs(result_path, exist_ok=True)
    
    else:
        srgan_train_dataset = getDataset(Config.train_patch_path_lr, Config.train_patch_path_hr, channels)
        srgan_train_loader = DataLoader(srgan_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        srgan_valid_dataset = getDataset(Config.valid_down_path_lr, Config.valid_down_path_hr, 3)
        srgan_valid_loader = DataLoader(srgan_valid_dataset, batch_size=1, shuffle=True)
        gen_model = Generator(Config.SCALE_FACTOR, channels).to(Config.device)
        disc_model = Discriminator(channels).to(Config.device)
        gen_optim = optim.Adam(gen_model.parameters(), lr=Config.LEARNING_RATE)
        disc_optim = optim.Adam(disc_model.parameters(), lr=Config.LEARNING_RATE)
        criterion = GeneratorCriterion().to(Config.device)
        ct = datetime.datetime.now()
        result_path = f'../output/Training/SRGAN/{ct.year}.{ct.month}.{ct.day}_{ct.hour}.{ct.minute}.{ct.second}'
        os.makedirs(result_path, exist_ok=True)

    snr_data, ssim_data = [],[]
    for epoch in range(1, Config.EPOCHS+1):
        print(f'epoch {epoch}/{Config.EPOCHS}')
        temp_result = None
        if arch == 'srcnn':
            train_srcnn(srcnn_model, srcnn_optim, criterion, srcnn_train_loader)
            temp_result = validate(srcnn_model, srcnn_valid_loader, channels)
        else:
            train_srgan(gen_model, disc_model, gen_optim, disc_optim, criterion, srgan_train_loader)
            temp_result = validate(gen_model, srgan_valid_loader, channels)
        
        print(f"PSNR = {temp_result[0]:.3f}, SSIM = {temp_result[1]:.3f}")
        snr_data.append(temp_result[0])
        ssim_data.append(temp_result[1])
        if (epoch % 100 == 0 or epoch==1):
            torch.save(srcnn_model.state_dict(), f"{result_path}/model.pth")
            saveAsCsv(snr_data, ssim_data, result_path)
            plot_snr(snr_data, result_path)
            plot_ssim(ssim_data, result_path)
            check_path = result_path + f'/sanity_checks'
            os.makedirs(check_path, exist_ok=True)
            sanity_check_dir(srcnn_model, Config.test_down_path_lr_up, Config.test_down_path_hr, check_path + f'/{epoch}.pdf', True, channels)


def train_srcnn(model, optim, criterion, loader):
    model.train()
    for _, data in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        low_res = data[0].to(Config.device)
        high_res = data[1].to(Config.device)
        pred = model(low_res)
        loss = criterion(pred, high_res)
        optim.zero_grad()
        loss.backward()
        optim.step()

def train_srgan(gen_model, disc_model, gen_optim, disc_optim, criterion, loader):
    gen_model.train()
    disc_model.train()
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        low_res = data[0].to(Config.device)
        high_res = data[1].to(Config.device)
        pred = gen_model(low_res)

        # Discriminator training
        disc_optim.zero_grad()
        real_output = disc_model(high_res).mean()
        fake_output = disc_model(pred).mean()
        disc_loss = 1 - real_output + fake_output
        disc_loss.backward(retain_graph=True)
        disc_optim.step()

        # Generator training
        disc_model.eval()
        gen_optim.zero_grad()
        fake_output = disc_model(pred).mean()
        gen_loss = criterion(pred, high_res, fake_output, Config.device)
        gen_loss.backward()
        gen_optim.step()
        disc_model.train()

def validate(model, loader, channels):
    model.eval()
    valid_psnr = 0.0
    valid_ssim = 0.0
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
            low_res = data[0].to(Config.device)
            high_res = data[1].to(Config.device)
            pred = None

            if channels == 1:
                pred_Y = model(low_res[:, 0, :, :])
                pred = torch.zeros((low_res.shape[0], 3, low_res.shape[2], low_res.shape[3]), dtype=torch.float)
                pred[:,0,:,:] = pred_Y
                pred[:,1,:,:] = low_res[:, 1, :, :]
                pred[:,2,:,:] = low_res[:, 2, :, :]
                pred = pred.to(Config.device)
            elif channels == 3:
                pred = model(low_res)
            valid_psnr += psnr(pred, high_res)
            valid_ssim += ssim(pred, high_res, data_range=1).item()
    return (valid_psnr / len(loader), valid_ssim / len(loader))


if __name__ == '__main__':
    main()
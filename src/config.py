import torch

class Config:
    "Contains all the necessary information and hyperparameters for training and testing."
    SCALE_FACTOR = 2
    EPOCHS = 2500
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    PATCH_SIZE = 64
    STRIDE = 28

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = '../data/Training'
    train_patch_path = '../data/Training_patches'
    train_patch_path_lr = f'../data/Training_patches/lr{SCALE_FACTOR}'
    train_patch_path_lr_up = f'../data/Training_patches/lr_up{SCALE_FACTOR}'
    train_patch_path_hr = f'../data/Training_patches/hr{SCALE_FACTOR}'

    valid_path = '../data/Validation'
    valid_down_path = '../data/Validation_downscaled'
    valid_down_path_lr = f'../data/Validation_downscaled/lr{SCALE_FACTOR}'
    valid_down_path_lr_up = f'../data/Validation_downscaled/lr_up{SCALE_FACTOR}'
    valid_down_path_hr = f'../data/Validation_downscaled/hr{SCALE_FACTOR}'

    test_path = '../data/Testing'
    test_down_path = '../data/Testing_downscaled'
    test_down_path_lr = f'../data/Testing_downscaled/lr{SCALE_FACTOR}'
    test_down_path_lr_up = f'../data/Testing_downscaled/lr_up{SCALE_FACTOR}'
    test_down_path_hr = f'../data/Testing_downscaled/hr{SCALE_FACTOR}'
import torch.nn as nn
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, VGG16_Weights

NUM_OF_RESIDUAL = 10
NUM_OF_DISC_BLOCKS = 7

class SuperResolutionCNN(nn.Module):
    "Super resolution convolutional neural network model."
    def __init__(self, channels):
        super(SuperResolutionCNN, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels = channels,
                        out_channels = 64,
                        kernel_size = 9,
                        stride = 1,
                        padding = 2
                    )
        self.conv2 = nn.Conv2d(
                        in_channels = 64,
                        out_channels = 32,
                        kernel_size = 1,
                        stride = 1,
                        padding = 2
                    )
        self.conv3 = nn.Conv2d(
                        in_channels = 32,
                        out_channels = channels,
                        kernel_size = 5,
                        stride = 1,
                        padding = 2
                    )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Generator(nn.Module):
    "Generator model from super resolution generative adversarial network."
    def __init__(self, scale_factor, channels):
        super(Generator, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(
                in_channels = channels,
                out_channels = 64,
                kernel_size = 9,
                stride = 1,
                padding = 4
            ),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(NUM_OF_RESIDUAL)]
        )
        self.norm_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample_block = nn.Sequential(
            *[UpsampleBlock(2) for _ in range(scale_factor // 2)]
        )
        self.last = nn.Conv2d(64, channels, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.first_block(x)
        y = self.residual_blocks(x)
        y = self.norm_block(y)
        x = self.upsample_block(x + y)
        x = self.last(x)
        return (torch.tanh(x) + 1) / 2

class Discriminator(nn.Module):
    "Discriminator model from super resolution generative adversarial network."
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.disc_blocks = nn.Sequential(
            ConvBlock(64, 64, 2),
            ConvBlock(64, 128, 1),
            ConvBlock(128, 128, 2),
            ConvBlock(128, 256, 1),
            ConvBlock(256, 256, 2),
            ConvBlock(256, 512, 1),
            ConvBlock(512, 512, 2),
        )
        self.final_blocks = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        size = x.size(0)
        x = self.first_block(x)
        x = self.disc_blocks(x)
        x = self.final_blocks(x)
        return torch.sigmoid(x.view(size))

class ResidualBlock(nn.Module):
    "Residual block from generator model."
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return (x + y)

class ConvBlock(nn.Module):
    "Convolutional block from discriminator model."
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):
    "Upsampling block from generator model."
    def __init__(self, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class GeneratorCriterion(nn.Module):
    "Loss function class for generator model."
    def __init__(self, channels):
        super(GeneratorCriterion, self).__init__()
        self.channels = channels
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        perception_net = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in perception_net.parameters():
            param.requires_grad = False
        self.perception_net = perception_net
        self.mse =  nn.MSELoss()

    def forward(self, predictions, high_res, fake_output, device):
        adversarial_loss = torch.mean(1 - fake_output)
        image_loss = self.mse(predictions, high_res)
        perception_loss = None

        if self.channels == 3:
            perception_loss = self.mse(self.perception_net(predictions.to(device)), self.perception_net(high_res.to(device)))
        else:
            pred_temp = torch.zeros(predictions.shape[0], 3, predictions.shape[2], predictions.shape[3])
            pred_temp[:,0,:,:] = predictions[:,0,:,:]
            high_temp = torch.zeros(high_res.shape[0], 3, high_res.shape[2], high_res.shape[3])
            high_temp[:,0,:,:] = high_res[:,0,:,:]
            perception_loss = self.mse(self.perception_net(pred_temp.to(device)), self.perception_net(high_temp.to(device)))

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss
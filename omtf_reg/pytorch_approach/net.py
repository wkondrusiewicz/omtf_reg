import torch
import torch.nn as nn
import torch.nn.functional as F


class omtfNet(nn.Module):
    def __init__(self):
        super(omtfNet, self).__init__()
        # self.conv1 = ConvBlock(1, 32, (6, 2), (2, 0))
        # self.conv2 = ConvBlock(32, 128, (3, 1), (1, 0))
        # self.conv3 = ConvBlock(128, 512, (2, 1))
        # self.conv4 = ConvBlock(512, 1024, (1, 1))

        self.conv1 = ConvBlock(1, 32, (3, 3), (1, 1))
        self.conv2 = ConvBlock(32, 64, (3, 2), (1, 0))
        self.conv3 = ConvBlock(64, 128, (3, 1), (1,0))
        self.conv4 = ConvBlock(128, 256, (3, 1))

        self.dense1 = DenseBlock(256, 4096)
        self.dense2 = DenseBlock(4096, 1024)
        self.dense3 = DenseBlock(1024, 256)
        self.dense4 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        # x = self.conv1(x)
        # x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        # x = self.conv2(x)
        # x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv4(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, padding: tuple = (0, 0)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            padding=padding), nn.BatchNorm2d(num_features=out_channels), nn.ELU())

    def forward(self, x):
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_last_layer: bool = False):
        super(DenseBlock, self).__init__()
        self.dense = nn.Sequential(nn.Linear(
            in_features=in_features, out_features=out_features), nn.ELU(), nn.Dropout()) if is_last_layer else nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.dense(x)
        return x

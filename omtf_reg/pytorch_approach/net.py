import torch
import torch.nn as nn
import torch.nn.functional as F


class omtfNetBigger(nn.Module):
    def __init__(self):
        super(omtfNetBigger, self).__init__()

        self.conv1 = ConvBlock(1, 64, (3, 1), (1, 0))
        self.conv2 = ConvBlock(64, 128, (3, 1), (1, 0))
        self.conv3 = ConvBlock(128, 256, (1, 2), (0, 0))
        self.conv4 = ConvBlock(256, 512, (3, 1), (1, 0))
        self.conv5 = ConvBlock(512, 1024, (3, 1), (1, 0))
        self.conv6 = ConvBlock(1024, 2048, (1, 1))

        self.dense1 = DenseBlock(2048, 4096)
        self.dense2 = DenseBlock(4096, 256)
        self.dense3 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv5(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv6(x)

        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class omtfNetBig(nn.Module):
    def __init__(self):
        super(omtfNetBig, self).__init__()

        self.conv1 = ConvBlock(1, 64, (3, 1), (1, 0))
        self.conv2 = ConvBlock(64, 128, (3, 1), (1, 0))
        self.conv3 = ConvBlock(128, 256, (3, 2), (1, 0))
        self.conv4 = ConvBlock(256, 512, (3, 1), (1, 0))
        self.conv5 = ConvBlock(512, 1024, (1, 1))

        self.dense1 = DenseBlock(1024, 4096, 0.2)
        self.dense2 = DenseBlock(4096, 256, 0.1)
        self.dense3 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv5(x)

        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class omtfNet(nn.Module):
    def __init__(self):
        super(omtfNet, self).__init__()

        self.conv1 = ConvBlock(1, 32, (5, 2), (2, 0))
        self.conv2 = ConvBlock(32, 64, (3, 1), (1, 0))
        self.conv3 = ConvBlock(64, 128, (3, 1), (1, 0))
        self.conv4 = ConvBlock(128, 256, (1, 1))

        self.dense1 = DenseBlock(256, 4096, 0)
        self.dense2 = DenseBlock(4096, 1024, 0)
        self.dense3 = DenseBlock(1024, 256, 0)
        self.dense4 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=(3, 1))(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x





class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple, padding: tuple = (0, 0)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding),
                                  # nn.BatchNorm2d(
                                  #     num_features=out_channels),
                                  nn.ELU())

    def forward(self, x):
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 dropout_probability: int = 0.5, is_last_layer: bool = False):
        super(DenseBlock, self).__init__()
        if is_last_layer:
            self.dense = nn.Linear(in_features=in_features,
                                   out_features=out_features)
        else:
            self.dense = nn.Sequential(nn.Linear(in_features=in_features,
                                                 out_features=out_features),
                #                        nn.BatchNorm1d(
                # num_features=out_features),
                nn.ELU(),
                nn.Dropout(p=dropout_probability))

    def forward(self, x):
        x = self.dense(x)
        return x


from functools import partial



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.activate = nn.ELU(inplace=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            self.activate,
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):

    def __init__(self, in_channels, out_channels, expansion=4, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=expansion, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             self.activate,
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             self.activate,
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class omtfResNet(nn.Module):
    """Inspired by https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278"""

    def __init__(self):
        super(omtfResNet, self).__init__()

        self.conv1 = Conv2dAuto(in_channels=1, out_channels=16, kernel_size=(3,3))
        self.res1 = ResNetBasicBlock(16, 64)
        self.res2 = ResNetBasicBlock(64, 128)
        self.res3 = ResNetBottleNeckBlock(128, 256, 2)

        self.dense1 = DenseBlock(512, 256)
        self.dense3 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = nn.MaxPool2d(kernel_size=(2,2))(x)
        x = self.res3(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense3(x)

        return x


class omtfResNetBig(nn.Module):
    def __init__(self):
        super(omtfResNetBig, self).__init__()

        self.conv1 = Conv2dAuto(in_channels=1, out_channels=16, kernel_size=(3,3))
        self.res1 = ResNetBasicBlock(16, 64)
        self.res2 = ResNetBasicBlock(64, 128)
        self.res3 = ResNetBottleNeckBlock(128, 256, 2)
        self.res5 = ResNetBottleNeckBlock(512, 512, 2)


        self.dense1 = DenseBlock(1024, 256)

        self.dense3 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = nn.MaxPool2d(kernel_size=(2,2))(x)
        x = self.res3(x)
        x = self.res5(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense3(x)

        return x


class omtfHalfResNet(nn.Module):
    def __init__(self):
        super(omtfHalfResNet, self).__init__()
        self.conv1 = Conv2dAuto(in_channels=1, out_channels=16, kernel_size=(3,3))
        self.res1 = ResNetBasicBlock(16, 32)
        self.res2 = ResNetBasicBlock(32, 64)
        self.res3 = ResNetBottleNeckBlock(64, 128, 2)
        self.conv2 = Conv2dAuto(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.conv3 = Conv2dAuto(in_channels=512, out_channels=1024, kernel_size=(3,1))

        self.dense1 = DenseBlock(1024, 256, 0)
        self.dense2 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.res3(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        return x


class omtfNetNotSoDense(nn.Module):
    def __init__(self):
        super(omtfNetNotSoDense, self).__init__()
        self.conv1 = Conv2dAuto(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = Conv2dAuto(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv3 = Conv2dAuto(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.conv4 = Conv2dAuto(in_channels=128, out_channels=256, kernel_size=(3,3))
        self.conv5 = Conv2dAuto(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.conv6 = Conv2dAuto(in_channels=512, out_channels=1024, kernel_size=(3,1))

        self.dense1 = DenseBlock(1024, 256, 0)
        self.dense2 = DenseBlock(256, 1, is_last_layer=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = nn.AdaptiveAvgPool2d((9,1))(x)
        x = self.conv6(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(-1, torch.prod(torch.tensor(x.size()[1:])))

        x = self.dense1(x)
        x = self.dense2(x)
        return x

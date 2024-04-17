from torch import nn

from clipmorph.nn.backbone import norm_batch_vgg


class FastStyleNet(nn.Module):
    def __init__(self):
        super(FastStyleNet, self).__init__()

        # Convolution block
        self.convBlock = nn.Sequential(
            ConvoLayer(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvoLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvoLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )

        # Residual block
        self.residualBlock = nn.Sequential(
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128)
        )

        # Deconvolution block
        self.convTransBlock = nn.Sequential(
            ConvTrans(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvTrans(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvoLayer(32, 3, kernel_size=9, stride=1, padding=4),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # TODO: Work only in 0.1:
        #     -> remove /255 in vgg norm thing
        #     -> change dataloader
        #     -> this should remove the *255 followed by the /255 between
        #           unet & vgg (i.e. vgg(unet(x)).
        #x = x / 255.0
        y = self.convBlock(x)
        y = self.residualBlock(y)
        y = self.convTransBlock(y)
        #y = self.sigmoid(y)
       # y = y * 255.0
        return y


class ConvoLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=1):
        super(ConvoLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        y = self.reflection_pad(x)
        y = self.conv2d(y)
        return y


class ResidualLayer(nn.Module):
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvoLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvoLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        y = y + residual
        return y


class ConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=1, output_padding=1):
        super(ConvTrans, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size, stride, padding,
                                            output_padding)

    def forward(self, x):
        return self.convTrans(x)
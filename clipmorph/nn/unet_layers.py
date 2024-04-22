from torch import nn

class ConvoLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
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
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        output_padding=1,
    ):
        super(ConvTrans, self).__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def forward(self, x):
        return self.convTrans(x)

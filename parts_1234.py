import torch
import torch.nn as nn
import torch.nn.functional as F

class two_conv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(two_conv, self).__init__()

        self.conv  = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv, self).__init__()
        self.conv = two_conv(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            two_conv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, 2, stride=2)

        self.conv = two_conv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class out(nn.Module):
    def __init__(self, in_channel, out_channel,):
        super(out, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


##Dense
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        #self.add_module('hardswish', nn.Hardswish())
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        x = super(_DenseBlock, self).forward(x)
        return x


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()


        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        #self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2))

    def forward(self, x):
        x = super(Transition, self).forward(x)
        return x


class indense(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, num_output_features):
        super(indense, self).__init__()
        num_features = num_input_features
        self.densein = torch.nn.Sequential()
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_input_features,
                            growth_rate=growth_rate)
        self.densein.add_module('denseblock', block)
        num_features = num_features + num_layers * growth_rate


        # trans = _Transition(num_input_features=num_features, num_output_features=num_output_features)
        # self.densein.add_module('trans', trans)

        # self.add_module('pool', nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        out = self.densein(x)
        return out


class dense_down(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, num_output_features):
        super(dense_down, self).__init__()
        num_features = num_input_features
        self.densedown = torch.nn.Sequential()
        self.densedown.add_module('pool', nn.MaxPool2d(kernel_size=2))

        block = _DenseBlock(num_layers=num_layers, num_input_features=num_input_features,
                            growth_rate=growth_rate)
        self.densedown.add_module('denseblock', block)

        num_features = num_features + num_layers * growth_rate
        # self.transtrans = _Transition(num_input_features=num_features, num_output_features=num_output_features)


    def forward(self, x):
        out = self.densedown(x)
        return out


class dense_up(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, num_output_features, bilinear=True):
        super(dense_up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(num_input_features // 2, num_input_features // 2, 2, stride=2)

        num_features = num_input_features
        self.denseup = torch.nn.Sequential()

        block = _DenseBlock(num_layers=num_layers, num_input_features=num_input_features,
                            growth_rate=growth_rate)
        self.denseup.add_module('denseblock', block)

        num_features = num_features + num_layers * growth_rate


    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        out = self.denseup(x)
        return out
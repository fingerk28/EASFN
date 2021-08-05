import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter


__all__ = ['Deasfn']


def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                     stride=stride, padding=(0, 1, 1), bias=False)


def ConvBlock(in_channels, out_channels, k, s):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )


class AttentionLayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=5):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()
        x = self.avg_pool(x)
        x = self.conv(x.squeeze(-1).transpose(-1, -2))

        return x


class ResidualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = conv1x3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Deasfn(nn.Module):
    def __init__(self):
        super(Deasfn, self).__init__()
        self.in_channels = 30

        self.FreqEncoder = nn.ModuleList()
        for i in range(1, 6):
            self.FreqEncoder.append(
                nn.Sequential(
                    nn.ConstantPad3d((0, 0, 0, 0, 5*i-1, 0), 0),
                    nn.Conv3d(30, 64, kernel_size=(5 * i, 1, 1), padding=0),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64,128, kernel_size=(25, 1, 1)),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True)
                )
            )

        self.SpatialEncoder = nn.Sequential(
            self.make_layer(ResidualBlock3D, 128, 4),
            nn.AvgPool3d((25, 1, 1))
        )

        self.hidden = Parameter(torch.randn(1, 1, (len(self.FreqEncoder)+1)*128))
        self.rnn = torch.nn.GRU(input_size=(len(self.FreqEncoder)+1)*128, hidden_size=(len(self.FreqEncoder)+1)*128)
        self.attention_block = AttentionLayer((len(self.FreqEncoder)+1)*128)

        self.decoder = nn.Sequential(
            ConvBlock((len(self.FreqEncoder)+1)*128, 79, 3, 2),
            ConvBlock(79, 38, 3, 1),
            ConvBlock(38, 19, 3, 1),
        )
        self.DecoderJH = nn.Sequential(
            ConvBlock(19, 19, 3, 1),
            ConvBlock(19, 19, 3, 1),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.DecoderPAF = nn.Sequential(
            ConvBlock(19, 38, 3, 1),
            ConvBlock(38, 38, 3, 1),
            nn.Conv2d(38, 38, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                conv3x3x3(self.in_channels, out_channels * block.expansion, stride=stride),
                nn.BatchNorm3d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, t, c, f, h, w = x.size()
        x = x.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

        # Spatial Encoder
        SpatialFeat = self.SpatialEncoder(x).squeeze(2)

        # Frequency Encoder
        FreqFeat = []
        for i in range(len(self.FreqEncoder)):
            FreqFeat.append(self.FreqEncoder[i](x).squeeze(2))
        FreqFeat = torch.cat(FreqFeat, dim=1)

        # Concatenate all the feature
        DualFeat = torch.cat([SpatialFeat, FreqFeat], dim=1)

        # Evolving attention module
        AttentionMask = []
        for i in range(t):
            AttentionMask.append(self.attention_block(DualFeat[i*b: (i+1)*b]).view(b, -1))
        attention = torch.stack(AttentionMask, dim=0)
        h0 = self.hidden.repeat(1, b, 1) 
        attention = self.rnn(attention, h0)[0]
        attention = attention.view(b*t, -1).unsqueeze(-1).unsqueeze(-1).expand_as(DualFeat)
        EvolvingFeat = DualFeat * attention

        # Decoder
        EvolvingFeat = F.interpolate(EvolvingFeat, size=[92, 124], mode='bilinear', align_corners=False)
        EvolvingFeat = self.decoder(EvolvingFeat)
        JH = self.DecoderJH(EvolvingFeat)
        PAF = self.DecoderPAF(EvolvingFeat)

        return JH, PAF


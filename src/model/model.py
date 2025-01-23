import torch
import torch.nn as nn
from src.Helpers.convBlock import ConvBlock
from src.Helpers.decoderBlock import DecoderBlock
from src.Helpers.encoderBlock import EncoderBlock

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, init_features)
        self.encoder2 = EncoderBlock(init_features, init_features * 2)
        self.encoder3 = EncoderBlock(init_features * 2, init_features * 4)
        self.encoder4 = EncoderBlock(init_features * 4, init_features * 8)

        self.bottleneck = ConvBlock(init_features * 8, init_features * 16)

        self.decoder4 = DecoderBlock(init_features * 16, init_features * 8)
        self.decoder3 = DecoderBlock(init_features * 8, init_features * 4)
        self.decoder2 = DecoderBlock(init_features * 4, init_features * 2)
        self.decoder1 = DecoderBlock(init_features * 2, init_features)

        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        x = self.final_conv(x)
        return torch.sigmoid(x)
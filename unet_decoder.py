import torch
import torch.nn as nn

from unet_layers import DoubleConv, Up


class UNetDecoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()

        # right now only b/w depth maps
        self.out_channels = 1

        layers = []
        feats = features_start*(2**(num_layers - 1))
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, self.out_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, xi):

        # up path / decoder
        for i, layer in enumerate(self.layers[:-1]):
            decoder_out = xi[-1]
            encoder_matching = xi[-2 - i]
            xi[-1] = layer(decoder_out, encoder_matching)

        # Final conv layer of UNet
        output = self.layers[-1](xi[-1])

        return output
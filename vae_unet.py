import torch
import torch.nn as nn
import torch.nn.functional as F

from data import NYUDepthDataModule
from unet_layers import DoubleConv, Up, Down


class VariationalUNet(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int = 3,
            enc_out_dim: int = 128,
            latent_dim: int = 128,
            kl_coeff: float = 0.01,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.kl_coeff = kl_coeff

        layers = [DoubleConv(self.input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, output_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)

        self.fc_mu = nn.Linear(1024*3*4, latent_dim)
        self.fc_logvar = nn.Linear(1024*3*4, latent_dim)

        self.projection_1 = nn.Linear(latent_dim, 1024*3*4)
        self.projection_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = x.squeeze(1)
        # concat x and y
        # x = torch.cat([x, y], dim=1)

        # down path / encoder
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.num_layers]:
            output = layer(xi[-1])
            xi.append(output)

        # embedding
        emb = xi[-1]
        emb = emb.view(emb.size(0), -1)

        # variational
        mu = self.fc_mu(emb)
        logvar = self.fc_logvar(emb)

        # kl
        z = self._reparameterize(mu, logvar)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

        # project emb and z to match original decoder dims
        first_dec_out = xi[-1]
        z = self.projection_1(z)
        z = z.view(first_dec_out.size())

        first_dec_out = torch.cat([first_dec_out, z], dim=1)
        first_dec_out = self.projection_2(first_dec_out)
        xi[-1] = first_dec_out

        # up path / decoder
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            decoder_out = xi[-1]
            encoder_matching = xi[-2 - i]
            xi[-1] = layer(decoder_out, encoder_matching)
        # Final conv layer of UNet
        output = self.layers[-1](xi[-1])

        return output, kl

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return z



#
# dm = NYUDepthDataModule('/Users/annikabrundyn/Developer/nyu_depth/data/', num_workers=0, resize=0.1, batch_size=2)
#
# in_channels = 3+1
#
# model = VariationalUNet(input_channels=in_channels, output_channels=1)
#
# img, target = next(iter(dm.train_dataloader()))
# img = img.squeeze(1)
# model(img, target)
#
# print("hey")
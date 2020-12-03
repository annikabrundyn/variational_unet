import torch
import torch.nn as nn
import torch.nn.functional as F

from data import NYUDepthDataModule
from unet_layers import DoubleConv, Up, Down
from unet_encoder import UNetEncoder
from unet_decoder import UNetDecoder


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

        self.encoder_x = UNetEncoder()
        self.encoder_xy = UNetEncoder(x_and_y=True)
        self.decoder = UNetDecoder()

        self.projection_1 = nn.Linear(latent_dim, 1024*3*4)
        self.projection_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = x.squeeze(1)

        enc_x_i, enc_x_mu, enc_x_logvar = self.encoder_x(x)
        enc_xy_i, enc_xy_mu, enc_xy_logvar = self.encoder_xy(x, y)

        # kl
        z = self._reparameterize(enc_xy_mu, enc_xy_logvar)
        kl = -0.5 * torch.sum(1 + enc_xy_logvar - enc_xy_mu.pow(2) - enc_xy_logvar.exp(), 1)

        # project emb and z to match original decoder dims
        z = self.projection_1(z)
        z = z.view(enc_x_i[-1].size())

        concat_enc_out_z = torch.cat([enc_x_i[-1], z], dim=1)
        concat_enc_out_z = self.projection_2(concat_enc_out_z)
        enc_x_i[-1] = concat_enc_out_z

        pred = self.decoder(enc_x_i)

        return pred, kl

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
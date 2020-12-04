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
            input_channels: int = 3,
            output_channels: int = 1,
            latent_dim: int = 128,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()

        self.encoder_x = UNetEncoder()
        self.encoder_xy = UNetEncoder(x_and_y=True)
        self.decoder = UNetDecoder()

        # TODO: remove hard coded dimensions
        self.projection_1 = nn.Linear(latent_dim, 1024*3*4)
        self.projection_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):

        enc_x_i, enc_x_mu, enc_x_logvar = self.encoder_x(x)
        enc_xy_i, enc_xy_mu, enc_xy_logvar = self.encoder_xy(x, y)

        # kl
        z = self._reparameterize(enc_xy_mu, enc_xy_logvar)
        kl = -0.5 * torch.sum(1 + enc_xy_logvar - enc_xy_mu.pow(2) - enc_xy_logvar.exp(), 1)

        # project z and concat with encoder output
        enc_x_i[-1] = self._project_and_concat(enc_x_i[-1], z)

        # decoder
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

    def _project_and_concat(self, enc_last_output, z):
        z = self.projection_1(z)
        z = z.view(enc_last_output.size())

        concat_enc_out_z = torch.cat([enc_last_output, z], dim=1)
        concat_enc_out_z = self.projection_2(concat_enc_out_z)

        return concat_enc_out_z



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
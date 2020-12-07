import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_encoder import UNetEncoder
from unet_decoder import UNetDecoder


class VariationalUNet(nn.Module):
    def __init__(
            self,
            resize: float,
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

        flat_enc_dim = self._calc_projection_dims(resize)

        self.fc_mu = nn.Linear(flat_enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_enc_dim, latent_dim)

        self.projection_1 = nn.Linear(latent_dim, flat_enc_dim)
        self.projection_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):

        enc_x_i = self.encoder_x(x)
        enc_xy_i = self.encoder_xy(x, y)

        # mu, logvar
        emb_xy = enc_xy_i[-1]
        emb_xy = emb_xy.view(emb_xy.size(0), -1)
        mu = self.fc_mu(emb_xy)
        logvar = self.fc_logvar(emb_xy)

        # kl
        z = self._reparameterize(mu, logvar)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

        # project z and concat with encoder output
        enc_x_i[-1] = self._project_and_concat(enc_x_i[-1], z)

        # decoder
        pred = self.decoder(enc_x_i)

        # feed prediction through sigmoid
        pred = F.sigmoid(pred)

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

    def _calc_projection_dims(self, resize):
        img_height = round(480 * resize)
        img_width = round(640 * resize)

        x = torch.rand((1, 3, img_height, img_width))

        enc_out = self.encoder_x(x)
        flat_dim = enc_out[-1].view(-1).shape[0]

        return flat_dim

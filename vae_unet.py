import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.fc_mu = nn.Linear(1024 * 12 * 24, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 12 * 24, latent_dim)

        # initialize weights to try prevent large std and nan values
        self.fc_mu.weight.data.uniform_(-0.01, 0.01)
        #self.fc_logvar.weight.data.uniform_(-0.01, 0.01)
        self.fc_logvar.weight.data.fill_(0)

        self.projection_1 = nn.Linear(latent_dim, 1024 * 12 * 24)
        self.projection_2 = nn.Sequential(
            nn.Conv2d(2 * 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def gaussian_like(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        log_pxz = log_pxz.sum(dim=(1,2,3))
        return log_pxz

    def forward(self, x, y):

        # concat x and y
        x = torch.cat([x, y], dim=1)

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

class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

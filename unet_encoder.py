import torch
import torch.nn as nn

from unet_layers import DoubleConv, Down


class UNetEncoder(nn.Module):
    def __init__(
            self,
            x_and_y: bool = False,
            latent_dim: int = 128,
            num_layers: int = 5,
            features_start: int = 64
    ):
        super().__init__()
        self.x_and_y = x_and_y
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        if self.x_and_y:
            self.input_channels = 3 + 1
        else:
            self.input_channels = 3

        layers = [DoubleConv(self.input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        self.layers = nn.ModuleList(layers)

        self.fc_mu = nn.Linear(1024*3*4, latent_dim)
        self.fc_logvar = nn.Linear(1024*3*4, latent_dim)


    def forward(self, x, y=None):

        x = x.squeeze(1)

        # concat x and y
        if self.x_and_y:
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

        return xi, mu, logvar
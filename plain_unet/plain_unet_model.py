from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.metrics.functional import ssim, psnr

from plain_unet.unet import UNet
from data import NYUDepthDataModule


class PlainUNet(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 1,
            input_channels: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            log_tb_imgs: bool = True,
            tb_img_freq: int = 5000,
            **kwargs
    ):

        super().__init__()

        self.save_hyperparameters()

        self.net = UNet(num_classes=1,
                        input_channels=3,
                        num_layers=num_layers,
                        features_start=features_start,
                        bilinear=bilinear)

    def forward(self, x):
        yhat = self.net(x).sigmoid()
        return yhat

    def step(self, batch):
        img, target = batch
        img = img.squeeze(1)
        pred = self(img)

        mse_loss = ((pred - target) ** 2).mean(dim=(1, 2, 3)).mean()

        ssim_val = ssim(pred, target)

        logs = {
            "mse_loss": mse_loss.mean(),
            "ssim": ssim_val,
        }

        img_logs = {
            "input": img[0].squeeze(0),
            "pred": pred[0],
            "target": target[0]
        }

        return mse_loss, logs, img_logs

    def training_step(self, batch, batch_idx):
        loss, logs, img_logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        if self.hparams.log_tb_imgs and batch_idx % self.hparams.tb_img_freq == 0:
            self.logger.experiment.add_image('train_input', img_logs['input'], self.trainer.global_step)
            self.logger.experiment.add_image('train_pred', img_logs['pred'], self.trainer.global_step)
            self.logger.experiment.add_image('train_target', img_logs['target'], self.trainer.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, img_logs = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        if self.hparams.log_tb_imgs and batch_idx % self.hparams.tb_img_freq == 0:
            self.logger.experiment.add_image('val_input', img_logs['input'], self.trainer.global_step)
            self.logger.experiment.add_image('val_pred', img_logs['pred'], self.trainer.global_step)
            self.logger.experiment.add_image('val_target', img_logs['target'], self.trainer.global_step)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [opt]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default='/Users/annikabrundyn/Developer/nyu_depth/data/', help="path to nyu depth data")
        parser.add_argument("--resize", type=float, default=1, help="percent to downsample images")
        parser.add_argument("--input_channels", type=int, default=1, help="number of frames to use as input")
        parser.add_argument("--num_classes", type=int, default=1, help="output channels")
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--log_tb_imgs", action='store_true', default=True)
        parser.add_argument("--tb_img_freq", type=int, default=5000)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")
        parser.add_argument("--num_workers", type=int, default=8)

        return parser


if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = PlainUNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = NYUDepthDataModule(
        args.data_dir,
        frames_per_sample=1,
        resize=args.resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_val=True
    )

    # model
    model = PlainUNet(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

import torch
import pytorch_lightning as pl

import os
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser

from data import NYUDepthDataModule
from vae_unet_model import VAEModel


if __name__ == "__main__":
    print("Predict")
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = VAEModel.add_model_specific_args(parser)
    parser.add_argument("--ckpt", required=True, type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    parser.add_argument("--model_name", required=True, type=str)
    args = parser.parse_args()

    #inputs_dir_path = os.path.join(args.output_dir, "inputs")
    #targets_dir_path = os.path.join(args.output_dir, "targets")
    pred_dir = os.path.join(args.output_dir, f"pred_{args.model_name}")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # data
    dm = NYUDepthDataModule(
        args.data_dir,
        frames_per_sample=1,
        resize=args.resize,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle_val=False,
    )

    # sanity check
    print("size of prediction samples:", len(dm.valset))

    # model
    model = VAEModel.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.to(device)
    model.eval()
    model.freeze()

    print("model instance created")
    print("lightning version", pl.__version__)

    outputs = []
    for idx, batch in enumerate(tqdm(dm.val_dataloader())):
        img, target = batch
        img = img.to(device)
        pred, _ = model(x=img)
        pred_imgs = [pred.squeeze(0)]

        # iterative refining of predictions
        for _ in range(model.refine_steps):
            pred, kl = model(x=img, y=pred)
            pred_imgs.append(pred.squeeze(0))

        torchvision.utils.save_image(pred_imgs[-1], fp=os.path.join(pred_dir, f"fin_pred_{idx}.png"))

        if idx % 500 == 0:
            pred_img_w_refine = torchvision.utils.make_grid(pred_imgs)
            torchvision.utils.save_image(pred_img_w_refine, fp=os.path.join(pred_dir, f"fin_pred_refine{idx}.png"))


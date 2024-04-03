from pathlib import Path
import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.autoencoder import get_model
from datasets import ImageNet, ImageNet100
from tqdm import tqdm
import argparse


torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256, bs=32):
    parser = argparse.ArgumentParser()
    # parser.add_argument("path", default="~/data/imagenet")
    args = parser.parse_args()
    args.path = "~/data/imagenet"
    args.path = Path(args.path).expanduser().resolve()
    args.in100_list = "~/lab/sgfm/scripts/imagenet100.txt"
    args.in100_list = Path(args.in100_list).expanduser().resolve()

    dataset = ImageNet100(
        path=args.path,
        in100_list=args.in100_list,
        resolution=resolution,
        random_flip=False,
    )
    train_dataset = dataset.get_split(split="train", labeled=True)
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = get_model("assets/stable-diffusion/autoencoder_kl.pth")
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # features = []
    # labels = []

    idx = 0

    for _, batch in enumerate(tqdm(train_dataset_loader)):
        img, label = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        img = img.to(device)
        moments = model(img, fn="encode_moments")
        moments = moments.detach().cpu().numpy()

        label = torch.cat([label, label], dim=0)
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(
                f"assets/datasets/imagenet100_{resolution}_features/{idx}.npy",
                (moment, lb),
            )
            idx += 1

    print(f"save {idx} files")


if __name__ == "__main__":
    main()

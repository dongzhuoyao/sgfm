import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm
import webdataset as wds
import sys


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    print(args)

    save_dir = f"assets/datasets/coco{resolution}_features/"
    if args.split == "train":
        datas = MSCOCODatabase(
            root="~/dataset/coco14/train2014",
            annFile="~/dataset/coco14/annotations/captions_train2014.json",
            size=resolution,
        )
    elif args.split == "val":
        datas = MSCOCODatabase(
            root="~/dataset/coco14/val2014",
            annFile="~/dataset/coco14/annotations/captions_val2014.json",
            size=resolution,
        )

    elif args.split == "debug":
        datas = MSCOCODatabase(
            root="~/dataset/coco14/val2014",
            annFile="~/dataset/coco14/annotations/captions_val2014.json",
            size=resolution,
        )
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model(
        "assets/stable-diffusion/autoencoder_kl.pth"
    )
    autoencoder.to(device)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    pattern = os.path.join(save_dir, f"coco14_{args.split}.tar")

    with wds.TarWriter(pattern) as sink:
        with torch.no_grad():
            for idx, data in tqdm(enumerate(datas)):
                x, captions = data

                if len(x.shape) == 3:
                    x = x[None, ...]
                x = torch.tensor(x, device=device)
                moments = autoencoder(x, fn="encode_moments").squeeze(0)
                moments = moments.detach().cpu().numpy()

                latent = clip.encode(captions)
                latent = latent.detach().cpu().numpy()

                sink.write(
                    {
                        "__key__": "sample%08d" % idx,
                        "image_feat.pyd": moments,
                        "caption_feat.pyd": latent,
                        "caption.pyd": captions,
                    }
                )

                if args.split == "debug" and idx > 1000:
                    break


if __name__ == "__main__":
    main()

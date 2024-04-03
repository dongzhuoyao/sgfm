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
from datasets import FeatureDataset_256_NPY
from tqdm import tqdm
import h5py


torch.manual_seed(0)
np.random.seed(0)


def main(
    debug=False,
    src_npy_path="/tmp/thu/imagenet256_features/",
    tgt_h5_path="/tmp/thu/imagenet256_features.h5",
    ds_name="imagenet256",
    batch_size=2048,
):
    # src_npy_path = Path(src_npy_path).expanduser().resolve()
    if debug:
        print("debug mode, save to debug.h5")
        tgt_h5_path = tgt_h5_path.replace(".h5", "_debug.h5")
    # tgt_h5_path = Path(tgt_h5_path).expanduser().resolve()

    dataset = FeatureDataset_256_NPY(
        path=src_npy_path,
        debug=debug,
    )

    train_dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    f = h5py.File(tgt_h5_path, mode="w")
    f.close()

    f = h5py.File(tgt_h5_path, mode="a")

    _len = len(train_dataset_loader.dataset)

    print("create dataset, length: ", _len)

    f.create_dataset(
        "train_feat", data=np.ones(shape=(_len, 8, 32, 32), dtype=np.float32) * -1
    )
    f.create_dataset("train_label", data=np.ones(shape=(_len, 1), dtype=np.int64) * -1)

    dset = f.create_dataset("all_attributes", (1,))
    dset.attrs["dataset_name"] = ds_name

    idx = 0
    for batch in tqdm(train_dataset_loader, desc="{}/{}".format(idx, _len)):
        img, label = batch
        for moment, lb in zip(img, label):
            f["train_feat"][idx, :] = moment
            f["train_label"][idx, :] = lb
            idx += 1

        if debug:
            print("debug mode, break")
            break

    print(f"save {idx} files")
    f.close()


if __name__ == "__main__":
    main()

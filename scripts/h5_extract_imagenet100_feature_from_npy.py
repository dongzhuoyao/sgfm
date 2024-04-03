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
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py


torch.manual_seed(0)
np.random.seed(0)


class FeatureDataset_NPY(Dataset):
    def __init__(self, path, debug=False):
        super().__init__()
        self.path = path
        names = os.listdir(path)

        self.files = [os.path.join(path, name) for name in names]
        self.debug = debug

        # test
        path = os.path.join(self.path, f"0.npy")
        z, label = np.load(path, allow_pickle=True)
        print(z.shape, z.dtype, label.shape, label.dtype)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        z, label = np.load(path, allow_pickle=True)
        return z, label


def main(
    debug=False,
    # src_npy_path="/tmp/thu/imagenet100_256_features/",
    # tgt_h5_path="/tmp/thu/imagenet100_256_features.h5",
    src_npy_path="/export/home/ra63nev/lab/sgfm/assets/datasets/imagenet100_256_features",
    tgt_h5_path="/export/home/ra63nev/lab/sgfm/assets/datasets/imagenet100_256_features.h5",
    ds_name="imagenet100_256",
    batch_size=128,
):
    # src_npy_path = Path(src_npy_path).expanduser().resolve()
    if debug:
        print("debug mode, save to debug.h5")
        tgt_h5_path = tgt_h5_path.replace(".h5", "_debug.h5")
    # tgt_h5_path = Path(tgt_h5_path).expanduser().resolve()

    dataset = FeatureDataset_NPY(
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

from pathlib import Path
import sys,os
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.autoencoder import get_model
from datasets import ImageNet
from tqdm import tqdm
import h5py 


torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256, debug=False, ds_name='imagenet256', save_flip = True, batch_size = 128 ):
    ds_path = "~/data/imagenet"
    ds_path = Path(ds_path).expanduser().resolve()
    feat_h5_path='~/lab/sgfm/assets/datasets/imagenet256_features.h5'
    if debug:
        print('debug mode, save to debug.h5')
        feat_h5_path = feat_h5_path.replace(".h5", "_debug.h5")
    feat_h5_path = Path(feat_h5_path).expanduser().resolve()
    ldm_encoder_decoder_path = '~/lab/sgfm/assets/stable-diffusion/autoencoder_kl.pth'
    ldm_encoder_decoder_path = Path(ldm_encoder_decoder_path).expanduser().resolve()
    

    dataset = ImageNet(
        path=ds_path, resolution=resolution, random_flip=False)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)


    model = get_model(ldm_encoder_decoder_path)
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)



    f = h5py.File(feat_h5_path, mode="w")
    f.close()

    f = h5py.File(feat_h5_path, mode="a")

    _len = len(train_dataset_loader.dataset)
    _len = _len * 2 if save_flip else _len
    
    print('create dataset, length: ', _len)


    f.create_dataset(
        "train_feat", data=np.ones(shape=(_len, 8, 32, 32), dtype=np.float32) * -1
    )
    f.create_dataset(
        "train_label", data=np.ones(shape=(_len, 1), dtype=np.int64) * -1
    )
    
    dset = f.create_dataset("all_attributes", (1,))
    dset.attrs["dataset_name"] = ds_name

    idx = 0
    for batch in tqdm(train_dataset_loader, desc='{}/{}'.format(idx, _len)):
        img, label = batch

        if save_flip:
            img = torch.cat([img, img.flip(dims=[-1])], dim=0)
            label = torch.cat([label, label], dim=0)

        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            f["train_feat"][idx, :] = moment
            f["train_label"][idx, :] = lb
            idx += 1

        if debug:
            print('debug mode, break')
            break

    print(f'save {idx} files')
    f.close()


if __name__ == "__main__":
    main()

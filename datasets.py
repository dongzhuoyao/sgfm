from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import h5py
from lfm_dataset.ffhq_from1024 import FFHQ_From1024
from lfm_dataset.real_img import Real_IMG
from pathlib import Path
from tqdm import tqdm
import os
import webdataset as wds

os.environ["WDS_VERBOSE_CACHE"] = "1"


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        assert x is not None
        assert y is not None
        return x, y


class DatasetFactory(object):
    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.0)
        v.clamp_(0.0, 1.0)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10


class CIFAR10(DatasetFactory):
    r"""CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None):
        super().__init__()

        transform_train = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        transform_test = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = datasets.CIFAR10(
            path, train=True, transform=transform_train, download=True
        )
        self.test = datasets.CIFAR10(
            path, train=False, transform=transform_test, download=True
        )

        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor(
            [len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]
        ).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f"{self.K} classes")
        print(f"cnt: {self.cnt}")
        print(f"frac: {self.frac}")

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def fid_stat(self):
        return "assets/fid_stats/fid_stats_cifar10_train_pytorch.npz"

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# ImageNet


class FeatureDataset_256_NPY(Dataset):  # used for h5_extract_npy_h5
    def __init__(self, path, debug=False):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]
        self.debug = debug

        # test
        path = os.path.join(self.path, f"0.npy")
        z, label = np.load(path, allow_pickle=True)
        print(z.shape, z.dtype, label.shape, label.dtype)

    def __len__(self):
        if self.debug:
            return 220_000
        else:
            return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        z, label = np.load(path, allow_pickle=True)
        return z, label


class FeatureDataset_IN_Cluster_H5(Dataset):
    def __init__(self, path, path_cluster, debug=False):
        super().__init__()
        self.path = path
        self.path_cluster = path_cluster
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]
        self.debug = debug
        print("Loading images from %s into memory..." % self.path_cluster)
        with h5py.File(self.path_cluster, "r") as f:
            if debug:
                self.cluster_assignment = f["cluster_assignment"][:10_000]  # [N, 1]
            else:
                self.cluster_assignment = f["cluster_assignment"][:]  # [N, 1
        print(
            "cluster_assignment",
            self.cluster_assignment.shape,
            self.cluster_assignment.dtype,
        )
        with h5py.File(self.path, "r") as f:
            if debug:
                self.train_feat = f["train_feat"][:10_000]  # [N, 1]
                self.train_label = f["train_label"][:10_000]  # [N, 1]
            else:
                self.train_feat = f["train_feat"][:]
                self.train_label = f["train_label"][:]

        assert len(self.train_feat) == len(self.train_label)
        self.len = len(self.train_feat)

    def __len__(self):
        if self.debug:
            return 2000
        else:
            return self.len - 10
            # return 253379 - 10  # consider the random flip

    def __getitem__(self, idx):
        z = self.train_feat[idx]
        cluster_id = self.cluster_assignment[idx]
        return z, cluster_id


class FeatureDataset_IN_Cluster_H5_lazyload(Dataset):  # a right writing for 70Gsize
    def __init__(self, path, path_cluster, debug=False):
        super().__init__()
        self.path = path
        self.path_cluster = path_cluster
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]
        self.debug = debug
        print("Loading images from %s, lazy mode!!..." % self.path_cluster)
        with h5py.File(self.path_cluster, "r") as f:
            if debug:
                self.cluster_assignment = f["cluster_assignment"][:10_000]  # [N, 1]
            else:
                self.cluster_assignment = f["cluster_assignment"][:]  # [N, 1
        print(
            "cluster_assignment",
            self.cluster_assignment.shape,
            self.cluster_assignment.dtype,
        )

        with h5py.File(self.path, "r") as f:
            self.len = len(f["train_feat"])
            print("train_feat", f["train_feat"].shape, f["train_label"].shape)
            assert len(f["train_feat"]) == len(f["train_label"])

    def __len__(self):
        if self.debug:
            return 2000
        else:
            return self.len - 10
            # return 253379 - 10  # consider the random flip

    def _open_hdf5(self):
        self._hf = h5py.File(self.path, "r")

    def __getitem__(self, idx):
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        _feat = np.array(self._hf["train_feat"][idx])
        # _label = int(self._hf["train_label"][idx])
        cluster_id = self.cluster_assignment[idx]
        return _feat, cluster_id


class FeatureDataset_Cluster_IN256(Dataset):
    def __init__(self, path, path_cluster, load_in_mem, debug=False):
        super().__init__()
        self.path = path
        self.path_cluster = path_cluster
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]
        self.debug = debug

        print("Loading images from %s into memory..." % self.path_cluster)
        with h5py.File(self.path_cluster, "r") as f:
            if debug:
                self.cluster_assignment = f["cluster_assignment"][:10_000]  # [N, 1]
            else:
                self.cluster_assignment = f["cluster_assignment"][:]  # [N, 1]

        with h5py.File(self.path_cluster, "r") as f:
            if debug:
                self.cluster_assignment = f["cluster_assignment"][:10_000]  # [N, 1]
            else:
                self.cluster_assignment = f["cluster_assignment"][:]  # [N, 1]

        print(
            "cluster_assignment",
            self.cluster_assignment.shape,
            self.cluster_assignment.dtype,
        )

    def __len__(self):
        if self.debug:
            return 10_000
        else:
            return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{str(idx).zfill(9)}.npy")
        if not os.path.isfile(path):
            path = os.path.join(self.path, f"{idx}.npy")

        try:
            z, _ = np.load(path, allow_pickle=True)
        except:
            z = np.load(path, allow_pickle=True)
        cluster_id = self.cluster_assignment[idx]
        # print(z.shape, z.dtype, cluster_id.shape, cluster_id.dtype)
        return z, cluster_id


class ChurchesFeatureDataset_H5(Dataset):
    def __init__(
        self,
        path,
    ):
        super().__init__()
        self.path = path
        print(f"loading, path: {path}")

        with h5py.File(self.path, "r") as f:
            self.len = len(f["train_feat"])
            print("dataset length", self.len)

    def __len__(self):
        return self.len

    def _open_hdf5(self):
        self._hf = h5py.File(self.path, "r")

    def __getitem__(self, idx):
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        _feat = np.array(self._hf["train_feat"][idx])
        return _feat


class ChurchesFeatureDataset_Cluster_H5(ChurchesFeatureDataset_H5):
    def __init__(self, path, path_cluster, load_in_mem=None, debug=False):
        super().__init__(path)
        self.path_cluster = path_cluster
        self.debug = debug

        print("Loading images from %s into memory..." % self.path_cluster)
        with h5py.File(self.path_cluster, "r") as f:
            self.cluster_assignment = f["cluster_assignment"][:]  # [N, 1]

        print(
            "cluster_assignment",
            self.cluster_assignment.shape,
            self.cluster_assignment.dtype,
        )

    def __getitem__(self, idx):
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        _feat = np.array(self._hf["train_feat"][idx])
        cluster_id = self.cluster_assignment[idx]
        return _feat, cluster_id


class ImageNetFeatures_H5(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, K=100, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        if path.endswith(".h5"):
            print("using h5 dataset")
            self.train = ImageNetFeatureDataset_H5(
                path=path, load_in_mem=True, debug=debug
            )
        else:
            raise
        print("Prepare dataset ok")
        self.K = K

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.K, (n_samples,), device=device)


class ImageNet256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        if path.endswith(".h5"):
            print("using h5 dataset")
            self.train = ImageNetFeatureDataset_H5(
                path=path, load_in_mem=True, debug=debug
            )
        else:
            raise NotImplementedError("should not use it any more ")
            print("using npy dataset")
            self.train = FeatureDataset_IN100(path=path, debug=debug)
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class IN_Features_EMA_H5(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, K=100, cfg=False, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        if path.endswith(".h5"):
            print("using h5 dataset")
            self.train = ImageNetFeatureDataset_H5(
                path=path, load_in_mem=True, debug=debug
            )
        else:
            raise NotImplementedError("should not use it any more ")
            print("using npy dataset")
            self.train = FeatureDataset_IN100(path=path, debug=debug)
        print("Prepare dataset ok")
        self.K = K

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.K, (n_samples,), device=device)


class ImageNet256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        self.train = FeatureDataset_IN100(path, debug=debug)
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/imagenet256_50k"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.K, (n_samples,), device=device)


class ImageNet256Features_Online_aaaa(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        self.train = FeatureDataset_Online(path, debug=debug)
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/imagenet256_50k"

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)

    @property
    def has_label(self):
        return True


class Churches256Features_Cluster(DatasetFactory):
    def __init__(
        self,
        path,
        path_cluster,
        cluster_k,
        cfg=False,
        p_uncond=None,
        load_in_mem=True,
        debug=False,
    ):
        super().__init__()
        print("Prepare dataset...")
        if True:
            self.train = ChurchesFeatureDataset_Cluster_H5(
                path, path_cluster=path_cluster, load_in_mem=load_in_mem, debug=debug
            )
        else:
            raise
        print("Prepare dataset ok")
        self.K = self.cluster_k = cluster_k
        print("*********  cfg", cfg)
        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/churches256_10k.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/churches256_10k"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.cluster_k, (n_samples,), device=device)


class ImageNet256Features_Cluster(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(
        self,
        path,
        path_cluster,
        cluster_k,
        cfg=False,
        p_uncond=None,
        load_in_mem=True,
        debug=False,
    ):
        super().__init__()
        print("Prepare dataset...")
        if True:
            self.train = FeatureDataset_Cluster_IN256(
                path, path_cluster=path_cluster, load_in_mem=load_in_mem, debug=debug
            )
        else:
            self.train = ImageNetFeatureDataset_H5(path, load_in_mem=False)
        print("Prepare dataset ok")
        self.K = self.cluster_k = cluster_k

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/imagenet256_50k"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.cluster_k, (n_samples,), device=device)


class IN_Features_Cluster_H5(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(
        self,
        path,
        path_cluster,
        cluster_k,
        cfg=False,
        p_uncond=None,
        debug=False,
    ):
        super().__init__()
        print("Prepare dataset...")

        self.train = FeatureDataset_IN_Cluster_H5_lazyload(
            path, path_cluster=path_cluster, debug=debug
        )

        print("Prepare dataset ok")
        self.K = self.cluster_k = cluster_k

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/imagenet256_50k"

    def sample_label(self, n_samples, device):
        return torch.randint(0, self.cluster_k, (n_samples,), device=device)


class ImageNet256Features_H5(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path_h5, cfg=False, p_uncond=None, debug=False):
        super().__init__()
        print("Prepare dataset...")
        if False:
            self.train = FeatureDataset(path_h5)
        else:
            self.train = ImageNetFeatureDataset_H5(
                path_h5, load_in_mem=False, debug=debug
            )
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = FeatureDataset(path)
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet100(DatasetFactory):
    def __init__(
        self, path, in100_list, resolution, random_crop=False, random_flip=True
    ):
        super().__init__()

        self.in100_list = in100_list
        with open(self.in100_list, "r") as f:
            self.in100_list = [_.strip() for _ in f.readlines()]
        if False:
            print(f"Counting ImageNet files from {path}")
            train_files = _list_image_files_recursively(os.path.join(path, "train"))
            class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        else:
            print(f"Counting ImageNet files from {path}")
            train_files = _list_image_files_recursively(os.path.join(path, "train"))
            class_names = [os.path.basename(path).split("_")[0] for path in train_files]
            train_files_new, class_names_new = [], []
            for tf, cn in zip(train_files, class_names):
                if cn in self.in100_list:
                    train_files_new.append(tf)
                    class_names_new.append(cn)

            train_files, class_names = train_files_new, class_names_new
            print("FFFFinish counting ImageNet files")
            print(len(train_files), len(class_names))
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print("Finish counting ImageNet files")

        self.train = ImageDataset(
            resolution,
            train_files,
            labels=train_labels,
            random_crop=random_crop,
            random_flip=random_flip,
        )
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f"Missing train samples: {len(self.train)} < 1281167")

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f"{self.K} classes")
        print(f"cnt[:10]: {self.cnt[:10]}")
        print(f"frac[:10]: {self.frac[:10]}")

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return (
            f"assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz"
        )

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f"Counting ImageNet files from {path}")
        train_files = _list_image_files_recursively(os.path.join(path, "train"))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print("Finish counting ImageNet files")

        self.train = ImageDataset(
            resolution,
            train_files,
            labels=train_labels,
            random_crop=random_crop,
            random_flip=random_flip,
        )
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f"Missing train samples: {len(self.train)} < 1281167")

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f"{self.K} classes")
        print(f"cnt[:10]: {self.cnt[:10]}")
        print(f"frac[:10]: {self.frac[:10]}")

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return (
            f"assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz"
        )

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class ImageNetFeatureDataset_H5(Dataset):
    def __init__(self, path, load_in_mem=True, debug=False):
        super().__init__()
        self.path = path
        print(f"loading, path: {path}")

        with h5py.File(self.path, "r") as f:
            self.len = len(f["train_feat"])
            print("dataset length", self.len)
            print("train_feat", f["train_feat"].shape, f["train_label"].shape)
            assert len(f["train_feat"]) == len(f["train_label"])

    def __len__(self):
        return self.len

    def _open_hdf5(self):
        self._hf = h5py.File(self.path, "r")

    def __getitem__(self, idx):
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        _feat = np.array(self._hf["train_feat"][idx])
        _label = int(self._hf["train_label"][idx])
        return _feat, _label


class CommonFeatureDataset_H5(Dataset):
    def __init__(self, path, np_num, load_in_mem=False, debug=False):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

        if load_in_mem:
            print("Loading images from %s into memory..." % path)
            with h5py.File(self.path, "r") as f:
                self._list = f["train_feat"][:]  # [N, 1]
        assert len(self._list) == self.np_num, f"{len(self._list)} != {self.np_num}"

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        z = self._list[idx]
        z = np.copy(z)
        return z, z


class CommonFeatureDataset_with_attr_H5(Dataset):
    def __init__(self, path, np_num, load_in_mem=False, debug=False):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

        if load_in_mem:
            print("Loading images from %s into memory..." % path)
            with h5py.File(self.path, "r") as f:
                self._list = f["train_feat"][:]  # [N, 1]
                self._attr = f["train_attr"][:]
        # assert len(self._list) == self.np_num, f"{len(self._list)} != {self.np_num}"

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        z = np.copy(self._list[idx])
        attr = np.copy(self._attr[idx])
        return z, attr


class CommonFeatureDataset(Dataset):
    def __init__(self, path, np_num):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        z, _ = np.load(path, allow_pickle=True)
        return z, z


class CommonFeatureDataset_Churches256(Dataset):
    def __init__(self, path, np_num):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{str(idx).zfill(9)}.npy")
        ##if not os.path.isfile(path):
        #  path = os.path.join(self.path, f"{idx}.npy")
        # try:
        #    z, _ = np.load(path, allow_pickle=True)
        # except:
        z = np.load(path, allow_pickle=True)
        return z, z


class CommonFeatureDataset_CM_Conditional(Dataset):
    def __init__(self, path, np_num):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        z, segmask, attr = np.load(path, allow_pickle=True)
        # return z, segmask, attr
        return z, attr


class CM256Features_Cond(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_CM_Conditional(path, np_num=30_001 - 1)
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class CM256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset(path, np_num=30_001 - 1)
        print("Prepare dataset ok")
        self.K = None

        if cfg:  # classifier free guidance
            raise NotImplementedError
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class FFHQ256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_with_attr_H5(
            path, np_num=60_000, load_in_mem=True
        )  # 60k-train,10k-val
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class MetFaces256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_H5(
            path, np_num=1336, load_in_mem=True
        )  # 60k-train,10k-val
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class Cat256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_H5(
            path, np_num=5065, load_in_mem=True
        )  # 60k-train,10k-val
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class Dog256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_H5(
            path, np_num=4678, load_in_mem=True
        )  # 60k-train,10k-val
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class Churches256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, **kwargs):
        super().__init__()
        print("Prepare dataset...")
        self.train = ChurchesFeatureDataset_H5(path)
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/churches256_10k.npz"

    @property
    def fid_stat_dir(self):
        return f"assets/fid_stats/churches256_10k"

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class CelebA(DatasetFactory):
    r"""train: 162,770
    val:   19,867
    test:  19,962
    shape: 3 * width * width
    """

    def __init__(self, path, resolution=64):
        super().__init__()

        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose(
            [
                Crop(x1, x2, y1, y2),
                transforms.Resize(self.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.train = datasets.CelebA(
            root=path, split="train", target_type=[], transform=transform, download=True
        )
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return "assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"

    @property
    def has_label(self):
        return False


class FFHQ(DatasetFactory):
    def __init__(self, path, resolution=256):
        super().__init__()

        self.train = FFHQ_From1024(
            root=path,
            size=resolution,
            split="train",
            random_crop=False,
            debug=False,
        )
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return "assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"

    @property
    def has_label(self):
        return False


class Real_IMG_DS(DatasetFactory):
    def __init__(self, path, resolution=256):
        super().__init__()

        self.train = Real_IMG(
            root=path,
            size=resolution,
            debug=False,
        )
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return "assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"

    @property
    def has_label(self):
        return False


# MS COCO


def center_crop(width, height, img):
    resample = {"box": Image.BOX, "lanczos": Image.LANCZOS}["lanczos"]
    crop = np.min(img.shape[:2])
    img = img[
        (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
        (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
    ]
    try:
        img = Image.fromarray(img, "RGB")
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MMCelebAHQ(Dataset):
    def __init__(
        self,
        root="~/data_hhd/mm-celeba-hq",
        size=256,
        is_for_dissect=False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.is_for_dissect = is_for_dissect

        self.size = size

        self.img_root = os.path.join(self.root, "image/images")
        self.caption_root = os.path.join(self.root, "text/celeba-caption")
        self._imgs = os.listdir(self.img_root)
        self._basenames = [x.split(".")[0] for x in self._imgs]
        self.filter_word = None  # "heavy makeup"
        if self.filter_word is not None:
            self._basenames = self.filter_by_word(self._basenames)
        print(f"MMCelebAHQ, Found {len(self._basenames)} images in {self.img_root}")

    def __len__(self):
        return len(self._basenames)

    def filter_by_word(self, basenames):
        filter_word = self.filter_word
        print("filter_by_word", filter_word, len(basenames))
        _basename_new = []
        for i, basename in tqdm(enumerate(basenames), total=len(basenames)):
            with open(os.path.join(self.caption_root, f"{basename}.txt"), "r") as f:
                captions = f.readlines()
                for _cap in captions:
                    if filter_word in _cap.lower().strip():
                        _basename_new.append(basename)

                        break
        print("filter_by_word", filter_word, len(_basename_new), len(basenames))
        return _basename_new

    def load_captions(self, index):
        with open(
            os.path.join(self.caption_root, f"{self._basenames[index]}.txt"), "r"
        ) as f:
            captions = f.readlines()
            captions = [x.strip() for x in captions if len(x.strip()) > 0]

        if self.filter_word is not None:
            _captions_new = []
            for _cap in captions:
                if self.filter_word in _cap.lower().strip():
                    _captions_new.append(_cap)
            captions = _captions_new

        return captions

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(self.img_root, f"{self._basenames[index]}.jpg")
        ).convert("RGB")

        image = np.array(image).astype(np.uint8)
        image = center_crop(self.size, self.size, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, "h w c -> c h w")

        _captions = self.load_captions(index)
        if not self.is_for_dissect:
            return image, _captions
        else:
            # always pick first caption for debuggin
            return image, _captions[0]


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None, is_for_dissect=False):
        from pycocotools.coco import COCO

        root = Path(root).expanduser().resolve()
        annFile = Path(annFile).expanduser().resolve()
        print("MSCOCODatabase", root, annFile)
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))
        self.is_for_dissect = is_for_dissect

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, "h w c -> c h w")

        if not self.is_for_dissect:
            anns = self._load_target(key)
            target = []
            for ann in anns:
                target.append(ann["caption"])
            return image, target
        else:
            anns = self._load_target(key)
            target = anns[0]["caption"]  # always pick first caption for debuggin
            return image, target


def get_feature_dir_info(root):
    print("get_feature_dir_info", root)
    files = glob.glob(os.path.join(root, "*.npy"))
    files_caption = glob.glob(os.path.join(root, "*_*.npy"))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in tqdm(files_caption, total=len(files_caption), desc="get_feature_dir_info"):
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split("_")
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.ds = wds.WebDataset(root)
        self.num_data = len(self.ds)
        print(f"MSCOCOFeatureDataset, root={root}, num_data={self.num_data}")
        print(self.ds[0].keys())

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if False:
            z = np.load(os.path.join(self.root, f"{index}.npy"))
            k = random.randint(0, self.n_captions[index] - 1)
            c = np.load(os.path.join(self.root, f"{index}_{k}.npy"))

        if False:
            if self.output_caption:
                with open(
                    os.path.join(self.root, f"{index}_{k}_captions.txt"), "r"
                ) as f:
                    captions = f.readlines()
                    captions = [x.strip() for x in captions if len(x.strip()) > 0]
                    assert self.n_captions[index] == len(
                        captions
                    ), f"{self.n_captions[index]} != {len(captions)}"

                return z, c, captions[k]

        return z, c


class MSCOCO256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None, output_caption=False):
        super().__init__()
        print("Prepare dataset...")

        self.train = MSCOCOFeatureDataset(
            path.replace("coco14_train.tar", "coco14_train.tar"),
        )

        self.test = MSCOCOFeatureDataset(
            path.replace("coco14_train.tar", "coco14_val.tar"),
        )

        self.empty_context = np.load(os.path.join(path, "empty_context.npy"))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(
            os.listdir(os.path.join(path, "run_vis")),
            key=lambda x: int(x.split(".")[0]),
        ):
            prompt, context = np.load(
                os.path.join(path, "run_vis", f), allow_pickle=True
            )
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_mscoco256_val.npz"


class MM_CelebA_HQ_FeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root, output_caption=False):
        self.root = root
        self.output_caption = output_caption
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f"{index}.npy"))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f"{index}_{k}.npy"))
        if self.output_caption:
            with open(os.path.join(self.root, f"{index}_{k}_captions.txt"), "r") as f:
                captions = f.readlines()
                captions = [x.strip() for x in captions if len(x.strip()) > 0]
                assert self.n_captions[index] == len(
                    captions
                ), f"{self.n_captions[index]} != {len(captions)}"

            return z, c, captions[k]

        return z, c


class MMCelebAHQ256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, output_caption=False):
        super().__init__()
        print("Prepare dataset...")

        self.train = MM_CelebA_HQ_FeatureDataset(
            os.path.join(path, "all"), output_caption=output_caption
        )

        self.test = self.train

        print("Prepare dataset ok")

        self.empty_context = np.load(os.path.join(path, "empty_context.npy"))

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(
            os.listdir(os.path.join(path, "run_vis")),
            key=lambda x: int(x.split(".")[0]),
        ):
            prompt, context = np.load(
                os.path.join(path, "run_vis", f), allow_pickle=True
            )
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_mscoco256_val.npz"


def get_dataset(name, **kwargs):
    if name == "cifar10":
        return CIFAR10(**kwargs)
    elif name == "imagenet":
        return ImageNet(**kwargs)
    elif name == "imagenet100_256_features":
        return ImageNetFeatures_H5(K=100, **kwargs)
    elif name == "imagenet100_256_features_ema":
        assert "K" in kwargs
        return IN_Features_EMA_H5(**kwargs)
    elif name == "imagenet100_256_features_cluster":
        return IN_Features_Cluster_H5(**kwargs)

    elif name == "imagenet256_features":
        return ImageNetFeatures_H5(K=1000, **kwargs)
    elif name == "imagenet256_features_cluster":
        return IN_Features_Cluster_H5(**kwargs)
    elif name == "imagenet256_features_ema":
        assert "K" in kwargs
        return IN_Features_EMA_H5(**kwargs)

    elif name == "churches256_features":
        return Churches256Features(**kwargs)
    elif name == "churches256_features_ema":
        return Churches256Features(**kwargs)
    elif name == "churches256_features_cluster":
        return Churches256Features_Cluster(**kwargs)

    elif name == "imagenet256_features_h5":
        raise NotImplementedError
        return ImageNet256Features_H5(**kwargs)
    elif name == "imagenet512_features":
        raise NotImplementedError
        return ImageNet512Features(**kwargs)

    elif name == "celeba":
        return CelebA(**kwargs)
    elif name == "celebamask256_features":
        return CM256Features(**kwargs)
    elif name == "celebamask256_features_cond":
        return CM256Features_Cond(**kwargs)
    elif name == "ffhq256_features":
        return FFHQ256Features(**kwargs)
    elif name == "metfaces256_features":
        return MetFaces256Features(**kwargs)
    elif name == "AFHQ256_cat_features":
        return Cat256Features(**kwargs)
    elif name == "AFHQ256_dog_features":
        return Dog256Features(**kwargs)
    elif name == "ffhq256":
        return FFHQ(**kwargs)

    elif name == "real_img":
        return Real_IMG_DS(**kwargs)

    elif name == "mscoco256_features":
        return MSCOCO256Features(**kwargs)
    elif name == "mscoco256":
        return MSCOCODatabase(**kwargs)

    elif name == "mmcelebahq256_features_withcaptioncontext":
        return MMCelebAHQ256Features(**kwargs)
    elif name == "mmcelebahq256_withcaptioncontext":
        return MMCelebAHQ(**kwargs)

    elif name == "mscoco256_features_withcaptioncontext":
        return MSCOCO256Features(output_caption=True, **kwargs)
    else:
        raise NotImplementedError(name)


def get_in_name_id_dict(path):
    train_files = _list_image_files_recursively(os.path.join(path, "train"))
    class_names = [os.path.basename(path).split("_")[0] for path in train_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    train_labels = [sorted_classes[x] for x in class_names]
    print("Finish counting ImageNet files")

    with open("my_imagenet_class_to_id.txt", "w") as f:
        f.write("\n".join([f"{k} {v}" for k, v in sorted_classes.items()]))

    with open("my_imagenet_path_label_list.txt", "w") as f:
        for _path, _label in zip(train_files, train_labels):
            f.write(f"{_path} {_label}\n")


if __name__ == "__main__":
    if False:
        ds = MMCelebAHQ()
        for _data in ds:
            print(_data[0].shape)
            print(_data[1])
            break
    else:
        ds_path = "~/data/imagenet"
        ds_path = Path(ds_path).expanduser().resolve()
        get_in_name_id_dict(ds_path)

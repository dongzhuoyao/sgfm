from collections import Counter
from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from clustering.cal_cluster_metric import cal_cluster_metric
from datasets import (
    Churches256Features_Cluster,
    IN_Features_Cluster_H5,
    ImageNet256Features_Cluster,
)
import libs
from tools.fid_score import calculate_fid_given_paths
from torchvision.utils import save_image, make_grid
import libs.autoencoder
import h5py
from torch.utils.data import Dataset
import wandb
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv(".env")
# Now you can access the variables using os.getenv
wandb_key = os.getenv("wandb_key")

autoencoder = libs.autoencoder.get_model("assets/stable-diffusion/autoencoder_kl.pth")
autoencoder.to("cuda")


class FeatureDataset_LabelCluster_IN100(Dataset):
    def __init__(self, path, path_cluster, debug=False):
        super().__init__()
        self.path = path
        self.path_cluster = path_cluster

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
            return 2000
        else:
            return 253379 - 10  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{str(idx).zfill(9)}.npy")
        if not os.path.isfile(path):
            path = os.path.join(self.path, f"{idx}.npy")

        z, label = np.load(path, allow_pickle=True)
        cluster_id = self.cluster_assignment[idx]

        return z, label, cluster_id


@torch.cuda.amp.autocast()
def encode(_batch):
    return autoencoder.encode(_batch)


@torch.cuda.amp.autocast()
def decode(_batch):
    return autoencoder.decode(_batch)


import fire


class MAIN(object):
    def __init__(self) -> None:
        pass

    def vis_churches_clusters(self, clusters=[i for i in range(20)], vis_num_min=16):
        from configs.churches256 import CONSTANT_CHURCHES256

        ds = Churches256Features_Cluster(
            path="assets/datasets/churches256_features",
            path_cluster=CONSTANT_CHURCHES256.layer10_t05_k100,
            cluster_k=100,
            cfg=False,
            p_uncond=0.15,
        )
        dl = DataLoader(
            ds.train, batch_size=512, shuffle=False, num_workers=8, pin_memory=False
        )

        clusters_dict = {k: [] for k in clusters}
        cluster_counter = Counter()

        for Xs, Ys in tqdm(
            dl,
            desc="collect",
        ):
            Xs = Xs.to("cuda")
            Ys = Ys.to("cuda")
            for _x, _y in zip(Xs, Ys):
                if _y.item() in clusters:
                    clusters_dict[_y.item()].append(_x)

        for k in clusters_dict:
            print("cluster {} has {} images".format(k, len(clusters_dict[k])))
            if len(clusters_dict[k]) == 0:
                print(
                    "cluster {} has {} images, less than 0, skip".format(
                        k, len(clusters_dict[k])
                    )
                )
                continue
            clusters_dict[k] = torch.stack(clusters_dict[k], dim=0)
            _z = autoencoder.sample(clusters_dict[k][:vis_num_min])
            trainbatch_4vis = decode(_z)
            trainbatch_4vis = (trainbatch_4vis + 1.0) / 2.0
            trainbatch_4vis = make_grid(trainbatch_4vis, 8)
            save_image(
                trainbatch_4vis, f"vis_cluster/churches_cluster_vis_{k}.png", nrow=8
            )

        print("done vis_cluster_images, exit(0)")

    def vis_imagenet_clusters(self, clusters=[i for i in range(20)], vis_num_min=16):
        ds = ImageNet256Features_Cluster(
            path="assets/datasets/imagenet256_features",
            path_cluster="assets/ssl_feats/imagenet256_features_dm_v3_layer10_t500.0_v0_cluster5000_iter30minp200_nns-1_2023-10-28T18_c6633e2.h5",
            cluster_k=5000,
            cfg=False,
            p_uncond=0.15,
        )
        dl = DataLoader(
            ds.train, batch_size=512, shuffle=False, num_workers=8, pin_memory=False
        )

        clusters_dict = {k: [] for k in clusters}
        cluster_counter = Counter()

        for Xs, Ys in tqdm(
            dl,
            desc="collect",
        ):
            Xs = Xs.to("cuda")
            Ys = Ys.to("cuda")
            for _x, _y in zip(Xs, Ys):
                if _y.item() in clusters:
                    clusters_dict[_y.item()].append(_x)

        for k in clusters_dict:
            print("cluster {} has {} images".format(k, len(clusters_dict[k])))
            if len(clusters_dict[k]) == 0:
                print(
                    "cluster {} has {} images, less than 0, skip".format(
                        k, len(clusters_dict[k])
                    )
                )
                continue
            clusters_dict[k] = torch.stack(clusters_dict[k], dim=0)
            _z = autoencoder.sample(clusters_dict[k][:vis_num_min])
            trainbatch_4vis = decode(_z)
            trainbatch_4vis = (trainbatch_4vis + 1.0) / 2.0
            trainbatch_4vis = make_grid(trainbatch_4vis, 8)
            save_image(
                trainbatch_4vis, f"vis_cluster/imagenet_cluster_vis_{k}.png", nrow=8
            )

        print("done vis_cluster_images, exit(0)")

    def cal_nmi(
        self,
    ):
        from configs.ablation import CONSTANT_IN100_FULL

        _datestr = datetime.now().strftime("%Y-%m-%d_%H_%M")
        wandb.login(relogin=True, key=wandb_key)
        wandb.init(project="cal_nmi", name=_datestr)
        print(CONSTANT_IN100_FULL.__dict__)
        name_dict = {
            i: CONSTANT_IN100_FULL.__dict__[i]
            for i in CONSTANT_IN100_FULL.__dict__
            if i.startswith("layer")
        }

        for _name, _path in tqdm(name_dict.items()):
            print(_name, _path)

        for _name, _path in tqdm(name_dict.items()):
            print(_name, _path)
            ds = FeatureDataset_LabelCluster_IN100(
                path="assets/datasets/imagenet100_256_features",
                path_cluster=_path,
                debug=False,
            )
            dl = DataLoader(
                ds, batch_size=50, shuffle=True, num_workers=8, pin_memory=True
            )

            try:
                label_list, cluster_list = [], []
                for i, (_, labels, clusterids) in enumerate(dl):
                    label_list.append(labels)
                    cluster_list.append(clusterids)

                label_list = torch.cat(label_list, dim=0).cpu().numpy()
                cluster_list = torch.cat(cluster_list, dim=0).cpu().numpy()
                nmi_dict = cal_cluster_metric(label_list, cluster_list)
                nmi_dict = {f"{_name}_{k}": v for k, v in nmi_dict.items()}
                print(nmi_dict)
                print("*" * 88)
                wandb.log(nmi_dict)
            except Exception as e:
                print(e)

    def check_file_exist(self):
        from configs.ablation import CONSTANT_IN100

        dirs = [
            CONSTANT_IN100.__dict__[i]
            for i in CONSTANT_IN100.__dict__
            if i.startswith("layer")
        ]
        print(dirs)
        for _file in dirs:
            if isinstance(_file, str):
                assert os.path.isfile(_file), _file

    def eval_genai(
        self,
        path_imgs="/tmp/tmpmkm15ani",
        fid_stat="assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz",
    ):
        from pytorch_fid import fid_score as sfid_score
        import torch_fidelity

        print(path_imgs)
        n_samples = len(os.listdir(str(path_imgs)))
        assert n_samples > 0
        print("n_samples", n_samples)
        _fid = calculate_fid_given_paths((fid_stat, path_imgs))
        if False:
            sfid = fid_score.main(
                _path1=path_imgs,
                _path2="/home/thu/data/sg_fid_eval/in32_4debug",
            )
            logging.info("sfid", sfid)
        wand_eval_dict = dict()
        for isc_splits in [1, 10]:
            tf_metrics_dict = torch_fidelity.calculate_metrics(
                input1=path_imgs,
                cuda=True,
                isc=True,
                isc_splits=isc_splits,
                verbose=False,
            )
            wand_eval_dict[f"is_tf_s{isc_splits}"] = tf_metrics_dict[
                "inception_score_mean"
            ]


        wand_eval_dict[f"fid{n_samples}"] = _fid
        print(wand_eval_dict)
        wand_eval_dict = {f"eval/{k}": v for k, v in wand_eval_dict.items()}
        wandb.log(wand_eval_dict)

    def test_label_bug(self, h5_path="assets/datasets/imagenet100_256_features.h5"):
        with h5py.File(h5_path, "r") as f:
            print(f.keys())
            print("len", len(f["train_label"]))
            print(f["train_label"][:10])
            print(f["train_label"][-10:])
            print(f["train_label"][:10].shape)
            print(f["train_label"][-10:].shape)
            print(f["train_label"].shape)
            print(f["train_label"][:10].dtype)
            print(f["train_label"][-10:].dtype)
            print(f["train_label"].dtype)

    def get_image_filenames_for_label(
        self, label="n01494475", synsets_filepath="ddib_in_misc/synset_words.txt"
    ):
        """
        Returns the validation files for images with the given label. This is a utility
        function for ImageNet translation experiments.
        :param label: an integer in 0-1000
        """
        # First, retrieve the synset word corresponding to the given label
        base_dir = os.getcwd()
        # synsets_filepath = os.path.join(base_dir, "evaluations", "synset_words.txt")
        synsets = [line.split()[0] for line in open(synsets_filepath).readlines()]
        synset_word_for_label = synsets[label]

        # Next, build the synset to ID mapping
        synset_mapping_filepath = os.path.join(
            base_dir, "evaluations", "map_clsloc.txt"
        )
        synset_to_id = dict()
        with open(synset_mapping_filepath) as file:
            for line in file:
                synset, class_id, _ = line.split()
                synset_to_id[synset.strip()] = int(class_id.strip())
        true_label = synset_to_id[synset_word_for_label]
        print("true_label", true_label)
        return true_label


if __name__ == "__main__":
    fire.Fire(MAIN)

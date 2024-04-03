from datetime import datetime
import shutil
import ml_collections
import numpy as np
import torch
from clustering.cal_cluster_metric import cal_cluster_metric
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import tools.utils_uvit as utils_uvit
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import itertools
from flow_matching import Flow_Matching
import hydra
from hydra.core.hydra_config import HydraConfig

from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


from dotenv import load_dotenv
from my_metrics import MyMetric

load_dotenv(".env")
wandb_key = os.getenv("wandb_key")

NMI_LEN = 100000


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.0)
    v.clamp_(0.0, 1.0)
    return v


def to255(v):
    return (v * 255.0).to(torch.uint8)


def has_label(config):
    if "imagenet" in config.dataset.name:
        return True
    elif "churches" in config.dataset.name:
        return False
    else:
        raise NotImplementedError


def is_patch_based(config):
    if "patch" in config.nnet.name:
        return True
    else:
        return False


def delete_ckpt(ckpt_root):
    dirs = os.listdir(ckpt_root)
    dirs = [int(d.split(".")[0]) for d in dirs]
    dirs = sorted(dirs)
    if len(dirs) > 1:
        for d in dirs[:-1]:
            try:
                ckpt_path = os.path.join(ckpt_root, f"{d}.ckpt")
                shutil.rmtree(ckpt_path)
                print("remove ckpts", ckpt_path)
            except Exception as e:
                print(e)


def get_dataloader(config):
    dataset = get_dataset(debug=config.is_debug, **config.dataset)
    train_dataset = dataset.get_split(
        split="train",
        labeled=True,
    )
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.dl.num_workers,
        pin_memory=True,
        # persistent_workers=True,
    )
    return train_dataset_loader


def init_wandb_logger_accelerator(accelerator, config):
    accelerate.utils.set_seed(config.seed, device_specific=True)
    print("config.train.batch_size", config.train.batch_size)
    logging.info(
        f"Process {accelerator.process_index} using device: {accelerator.device}"
    )

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)

    if accelerator.is_main_process:
        wandb.login(relogin=True, key=wandb_key)
        wandb.init(
            reinit=True,
            dir=os.path.abspath(config.workdir),
            project="sgfm",
            config=config.to_dict(),
            name=config.job_name,
            job_type="train",
            mode="online",  # default offline
        )
        utils_uvit.set_logger(
            log_level="info", fname=os.path.join(config.workdir, "output.log")
        )
        logging.info(config)
    else:
        utils_uvit.set_logger(log_level="error")
        builtins.print = lambda *args: None
    logging.info(f"Run on {accelerator.num_processes} devices")


def train(config):
    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # mp.set_start_method("spawn"), this will cause error in mvl-gpu, comment it

    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    device = accelerator.device

    config = ml_collections.FrozenConfigDict(config)
    sigma_min = config.dynamic.sigma_min

    accelerator.wait_for_everyone()
    init_wandb_logger_accelerator(accelerator, config)

    train_dl = get_dataloader(config)

    train_state = utils_uvit.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dl = accelerator.prepare(
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_dl,
    )
    lr_scheduler = train_state.lr_scheduler

    if len(os.listdir(config.ckpt_root)):
        train_state.resume(config.ckpt_root)
    else:
        if config.pretrained_path is None or len(config.pretrained_path) == 0:
            logging.warning("pretrained_path is None, will train from scratch")
        else:
            train_state.load_nnet_sgfm(config.pretrained_path)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(
                train_dl,
                disable=not accelerator.is_main_process,
            ):
                yield data

    train_dg = get_data_generator()

    fixed_noise = torch.randn(
        (
            config.vis_num * torch.cuda.device_count(),
            config.z_shape[0],
            config.z_shape[1],
            config.z_shape[1],
        ),
        device=device,
    )

    score_model = Flow_Matching(net=nnet)
    score_model_ema = Flow_Matching(net=nnet_ema)

    label_queue = []
    cluster_queue = []
    counter_clusterids = np.array([0 for _ in range(config.dataset.K)])
    fid_best = 666
    my_metric = MyMetric(
        choices=["fid", "is", "kid", "prdc", "sfid", "fdd"],
    )

    def train_step(
        _batch,
        cluster_queue,
        label_queue,
    ):
        _metrics = dict()
        optimizer.zero_grad()
        if has_label(config):
            _image, label = _batch
        else:
            _image = _batch

        x = (
            autoencoder.sample(_image)
            if "feature" in config.dataset.name
            else encode(_image)
        )
        noise = torch.randn_like(x)
        if train_state.step < config.ema_start_iters:
            is_stage2, y_emb = False, None
        else:
            is_stage2 = True
            ema_t = torch.tensor(
                [config.ema_t] * len(x), device=x.device, dtype=x.dtype
            )
            ema_t_4d = ema_t[:, None, None, None]  # [B, 1, 1, 1]
            ema_x_t = ema_t_4d * x + (1 - (1 - sigma_min) * ema_t_4d) * noise
            y_emb, dm_emb, cluster_ids = score_model_ema.get_cluster_info(
                x_t=ema_x_t, t=ema_t
            )
            y_emb = y_emb.detach()

        t = torch.rand(len(x), device=x.device, dtype=x.dtype)
        t_4d = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = t_4d * x + (1 - (1 - sigma_min) * t_4d) * noise
        u = x - (1 - sigma_min) * noise

        (_loss_fm, _softlabel) = score_model.get_loss_fm(
            x_t=x_t,
            u=u,
            t=t,
            y=y_emb,
            is_stage2=is_stage2,
            config=config,
        )

        if True:
            _loss_sk, cluster_ids = score_model.get_loss_sk(
                _softlabel=_softlabel,
                accelerator=accelerator,
            )
            if not is_patch_based(config):
                bool_flag = (t > 0.8) & (t < 0.9)
            else:
                t_latent = einops.rearrange(
                    t.repeat(1, 16 * 16), "b c->(b c)"
                )  # why is 16? the patch size is 32x32
                bool_flag = (t_latent > 0.8) & (t_latent < 0.9)
            _loss_sk = (_loss_sk * bool_flag).mean()
            _loss_sk = _loss_sk.mean()
            sk_warmup = min(train_state.step * 1.0 / config.ema_start_iters, 1)
            _loss = _loss_fm.mean() + _loss_sk * config.swav_w * sk_warmup

        accelerator.backward(_loss.mean())
        optimizer.step()
        lr_scheduler.step()

        _metrics["is_stage2"] = int(is_stage2)
        _metrics["loss_sk"] = _loss_sk.item()
        _metrics["loss_fm"] = _loss_fm.detach().mean().item()
        _metrics["sk_warmup"] = sk_warmup
        _metrics["loss"] = accelerator.gather(_loss.detach()).mean().item()
        _metrics["lr"] = train_state.optimizer.param_groups[0]["lr"]

        if has_label(config):
            label_queue.extend([_l.item() for _l in label])
        if True:
            cluster_queue.extend([_l.item() for _l in cluster_ids])
            if len(label_queue) > NMI_LEN:
                label_queue = label_queue[-NMI_LEN:]
                cluster_queue = cluster_queue[-NMI_LEN:]
            counter_clusterids[cluster_ids.cpu().numpy()] += 1

        if train_state.step == config.ema_start_iters:
            nnet_ema.load_state_dict(nnet.state_dict())
            print("loading nnet_ema from previous-stage ckpt")
            print("*" * 66)

        if train_state.step >= config.ema_start_iters:
            train_state.ema_update(config.get("ema_rate", 0.9999))
        train_state.step += 1
        return (
            _metrics,
            cluster_queue,
            label_queue,
            cluster_ids,
            t,
        )

    assert config.vis_num % 4 == 0

    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(train_dg))
        metrics, cluster_queue, label_queue, cluster_ids, t = train_step(
            batch, cluster_queue=cluster_queue, label_queue=label_queue
        )
        nnet.eval()

        if (
            accelerator.is_main_process
            and train_state.step % config.train.log_interval == 0
        ):
            if has_label(config):
                if len(label_queue) >= NMI_LEN:
                    metrics.update(
                        cal_cluster_metric(
                            gt_np=np.array(label_queue), pred_np=np.array(cluster_queue)
                        )
                    )

            print(metrics)
            logging.info(
                f" step={train_state.step}/{config.train.n_steps}, {config.workdir}"
            )
            if cluster_ids is not None:
                data = [[x, y] for (x, y) in zip(t, cluster_ids)]
                table = wandb.Table(data=data, columns=["timestep", "cluster_argmax"])
                metrics["time_vs_cluster"] = wandb.plot.scatter(
                    table, "timestep", "cluster_argmax"
                )
            wandb.log(
                metrics,
                step=train_state.step,
            )

        if (
            accelerator.is_main_process
            and train_state.step % config.train.vis_interval == 0
        ):
            torch.cuda.empty_cache()
            logging.info("Save a grid of images...")

            z = score_model.decode(
                fixed_noise[: config.vis_num], y=None,
            )

            if has_label(config):
                batch = batch[0]  # we always return y for debug purpose

            fake_raw = z

            real_raw = (
                autoencoder.sample(batch)
                if "feature" in config.dataset.name
                else encode(batch)
            )
            real_4vis = decode(real_raw[: config.vis_num])
            real_4vis = make_grid(unpreprocess(real_4vis), 10)

            fake_4vis = decode(z[: config.vis_num])
            fake_4vis = make_grid(unpreprocess(fake_4vis), 10)
            save_image(
                fake_4vis, os.path.join(config.sample_dir, f"{train_state.step}.png")
            )
            wandb.log(
                {
                    "vis/samples": wandb.Image(fake_4vis),
                    "vis/data": wandb.Image(real_4vis),
                    "data_range/gen_mean": fake_raw.mean(),
                    "data_range/gen_std": fake_raw.std(),
                    "data_range/gen_max": fake_raw.max(),
                    "data_range/gen_min": fake_raw.min(),
                    "data_range/data_mean": real_raw.mean(),
                    "data_range/data_std": real_raw.std(),
                    "data_range/data_min": real_raw.min(),
                    "data_range/data_max": real_raw.max(),
                },
                step=train_state.step,
            )
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if (
            train_state.step % config.train.save_interval == 0
            or train_state.step == config.train.n_steps
        ):
            torch.cuda.empty_cache()
            logging.info(f"Save and eval checkpoint {train_state.step}...")
            if (
                accelerator.local_process_index == 0
            ):  # cannot set it to is_main_process, if you are on multi-node
                if True:
                    data = [[s] for s in counter_clusterids]
                    table = wandb.Table(data=data, columns=["cluster_ids"])
                    wandb.log(
                        {
                            "counter_clusterids": wandb.plot.histogram(
                                table,
                                "cluster_ids",
                                title="clusterids assignments",
                            ),
                        },
                        step=train_state.step,
                    )
                    counter_clusterids = np.array([0 for _ in range(config.dataset.K)])

                if not config.save_every_ckpt:
                    delete_ckpt(config.ckpt_root)
                else:
                    print("save_every_ckpt is True, will not remove ckpts")
                train_state.save(
                    os.path.join(config.ckpt_root, f"{train_state.step}.ckpt")
                )

            accelerator.wait_for_everyone()
            print("evaluate fid...")

            if train_state.step < config.ema_start_iters:
                print("apply unconditional sampling")
            else:
                print(
                    f"apply conditional sampling with CFG, with {config.sample.scale}"
                )

            def cfg_nnet(x, timesteps, y, **kwargs):
                _cond = nnet(x, timesteps, y=y, **kwargs)
                if isinstance(_cond, tuple):
                    _cond = _cond[0]
                _uncond = nnet(x, timesteps, y=None, **kwargs)
                if isinstance(_uncond, tuple):
                    _uncond = _uncond[0]
                return _cond + config.sample.scale * (_cond - _uncond)

            score_model_eval_cfg = Flow_Matching(net=cfg_nnet)
            score_model_eval = Flow_Matching(net=nnet)

            def sample_fn(_n_samples):
                _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
                _batch = tree_map(lambda x: x.to(device), next(train_dg))
                if has_label(config):
                    _image, _ = _batch
                else:
                    _image = _batch
                if config.is_debug:
                    _n_samples = min(_n_samples, len(_image))
                else:
                    assert _n_samples <= len(_image)
                _image = _image[:_n_samples]
                x_train = (
                    autoencoder.sample(_image)
                    if "feature" in config.dataset.name
                    else encode(_image)
                )
                if train_state.step < config.ema_start_iters:
                    kwargs = dict(y=None, is_stage2=False)
                    _feat = score_model_eval.decode(
                        _z_init,
                        **kwargs,
                    )
                else:
                    sigma_min = config.dynamic.sigma_min
                    noise = torch.randn_like(x_train)
                    ema_t = torch.tensor([config.ema_t] * _n_samples, device=device)
                    ema_t_ = ema_t[:, None, None, None]  # [B, 1, 1, 1]
                    ema_x_t = ema_t_ * x_train + (1 - (1 - sigma_min) * ema_t_) * noise

                    y_cond, dm_emb, _cluster_ids = score_model_ema.get_cluster_info(
                        x_t=ema_x_t, t=ema_t
                    )

                    kwargs = dict(y=y_cond, is_stage2=True)
                    _feat = score_model_eval_cfg.decode(
                        _z_init,
                        **kwargs,
                    )

                return to255((unpreprocess(decode(x_train)))), to255(
                    (unpreprocess(decode(_feat)))
                )

            my_metric = utils_uvit.sample_torchmetrics(
                my_metric,
                accelerator,
                min(1e10, config.sample.n_samples),
                config.train.batch_size,
                sample_fn,
            )

            _metric_dict = my_metric.compute()
            fid = _metric_dict["fid"]
            _metric_dict = {f"eval/{k}": v for k, v in _metric_dict.items()}
            fid_best = min(fid, fid_best)

            if accelerator.is_main_process:
                print(f"FID: {fid}, best_fid: {fid_best}")
                if True:
                    wandb_dict = {
                        f"best_fid{my_metric._fid.fake_features_num_samples}": fid_best,
                        "real_num": my_metric._fid.real_features_num_samples,
                        "fake_num": my_metric._fid.fake_features_num_samples,
                    }
                    wandb_dict.update(_metric_dict)
                    wandb.log(
                        wandb_dict,
                        step=train_state.step,
                    )

            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()



@hydra.main(config_path="configs_hydra", config_name="default", version_base=None)
def main(config):
    version_str = "v2.9"

    if config.is_debug:
        config.train.n_steps = 110
        config.train.batch_size = 120
        # config.train.vis_interval = 1_000_000_000
        config.train.vis_interval = 50
        config.train.save_interval = 60
        config.train.log_interval = 1
        config.sample.n_samples = 400
        config.sample.mini_batch_size = 10
        config.scratch = True
        config.ema_start_iters = 0.5
        print("debug mode ******")

    assert isinstance(config.ema_start_iters, float) and config.ema_start_iters < 1.0
    config.ema_start_iters = config.ema_start_iters * config.train.n_steps
    print("ema_start_iters will start at", config.ema_start_iters)

    _swav_desc = (
        f"proK{config.nnet.nmb_prototypes}dim{config.nnet.nmb_prototypes_output_dim}"
    )
    _ema_desc = (
        f"uncond{config.nnet.p_uncond}_ema{config.ema_start_iters}_swav{config.swav_w}"
    )

    job_name = "_".join(
        [
            config.tag,
            f"d{int(config.is_debug)}",
            version_str,
            config.nnet.name,
            config.dataset.name,
            _ema_desc,
            _swav_desc,
        ]
    )

    config.job_name = job_name
    config.workdir = os.path.join("workdir", job_name)
    if not config.scratch:
        print("non-scratch mode...")
    else:
        print("from_scratch mode....")
        config.workdir = (
            config.workdir + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        print("new path", config.workdir)
        os.makedirs(config.workdir, exist_ok=True)

    config.ckpt_root = os.path.join(config.workdir, "ckpts")
    config.sample_dir = os.path.join(config.workdir, "samples")

    train(config)


if __name__ == "__main__":
    main()

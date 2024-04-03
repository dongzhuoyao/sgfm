from datetime import datetime
import os
from einops import rearrange, repeat
from tqdm import tqdm
import wandb
from flow_matching import Flow_Matching
from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import tools.utils_uvit as utils_uvit
from datasets import get_dataset
import tempfile
from absl import logging
import builtins
import libs.autoencoder
import torch_fidelity
from torchvision.utils import save_image, make_grid
import wandb
from torch.utils._pytree import tree_map
from absl import flags
from absl import app
from ml_collections import config_flags
import os

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv(".env")
# Now you can access the variables using os.getenv
wandb_key = os.getenv("wandb_key")


def vis_cfg(
    nnet,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
):
    cfg_range_num = 8
    cfg_strength_max = 10

    def cfg_nnet(x, timesteps, y, **kwargs):
        assert y is not None
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=torch.tensor([dataset.K] * x.size(0), device=device),
            **kwargs,
        )[0]
        assert len(x) == cfg_range_num
        scale = torch.linspace(0, cfg_strength_max, cfg_range_num).to(device)
        scale = scale.view(-1, 1, 1, 1)
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(1, *config.z_shape, device=device)
        _z_init = _z_init.repeat(_n_samples, 1, 1, 1)
        y = dataset.sample_label(1, device=device)
        y = y.repeat(_n_samples)
        if config.train.mode == "uncond":
            kwargs = dict()
        elif config.train.mode == "cond":
            kwargs = dict(y=y)
        else:
            raise NotImplementedError

        _feat = score_model.decode(
            _z_init,
            **kwargs,
        )
        return decode_large_batch(_feat)

    os.makedirs(sample_path, exist_ok=True)

    for exp_id in tqdm(range(10), "exploring 10 exps of cfg"):
        sampled_imgs = sample_fn(cfg_range_num)
        sampled_imgs = dataset.unpreprocess(sampled_imgs)
        _date_str = datetime.now().isoformat(timespec="hours")
        save_image(
            sampled_imgs,
            os.path.join(sample_path, f"sampled_imgs_{_date_str}_{exp_id}.png"),
        )


def vis_fm_chain(
    nnet,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
    img_num=16,
):
    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=torch.tensor([dataset.K] * x.size(0), device=device),
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
        rf_kwargs=config.rf_kwargs,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if config.train.mode == "uncond":
            kwargs = dict()
        elif config.train.mode == "cond":
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        chains = score_model.decode_fm_chain(
            _z_init,
            step_size=0.01,
            **kwargs,
        )
        img_gens = []
        for chain in chains:
            img_gens.append(decode_large_batch(chain).unsqueeze(0))
        img_gens = torch.cat(img_gens, dim=0)
        img_gens = rearrange(img_gens, "b n c h w -> (b n) c h w")
        return img_gens

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs = sample_fn(img_num)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)
    sampled_imgs = rearrange(sampled_imgs, "(b n) c h w -> b n c h w", b=img_num)
    for idx, sampled_img in enumerate(sampled_imgs):
        save_image(
            sampled_img,
            os.path.join(sample_path, f"vis_fm_chain_s{config.sample.scale}_{idx}.png"),
        )


def vis_cluster_samples_online(
    nnet,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
    img_num=8,
):
    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=torch.tensor([dataset.K] * x.size(0), device=device),
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        y = dataset.sample_label(_n_samples // 4, device=device)
        try:
            _prototype = nnet.module.prototypes.weight.data.clone()  # [100,1024]
        except:
            _prototype = nnet.prototypes.weight.data.clone()
        y = _prototype[y].view(_n_samples, -1).to(device)
        y = repeat(y, "b c -> (4 b) c")
        _feat = score_model.decode(
            _z_init,
            y=y,
        )
        return decode_large_batch(_feat)

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs = sample_fn(img_num)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)  # b c h w

    wandb.log(
        {
            "vis_cluster_samples_online": [
                wandb.Image(sampled_imgs, caption="vis_cluster_samples_online")
            ]
        }
    )


def vis_cluster_samples_online(
    nnet,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
    img_num=8,
):
    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=torch.tensor([dataset.K] * x.size(0), device=device),
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        y = dataset.sample_label(_n_samples // 4, device=device)
        try:
            _prototype = nnet.module.prototypes.weight.data.clone()  # [100,1024]
        except:
            _prototype = nnet.prototypes.weight.data.clone()
        y = _prototype[y].view(_n_samples, -1).to(device)
        y = repeat(y, "b c -> (4 b) c")
        _feat = score_model.decode(
            _z_init,
            y=y,
        )
        return decode_large_batch(_feat)

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs = sample_fn(img_num)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)  # b c h w

    wandb.log(
        {
            "vis_cluster_samples_online": [
                wandb.Image(sampled_imgs, caption="vis_cluster_samples_online")
            ]
        }
    )


def eval_cluster_vis_during_training(
    nnet,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
    img_num=128,
    seed=46,
):
    utils_uvit.set_seed(
        seed,
    )

    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=None,
            device=device,
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        y = dataset.sample_label(_n_samples // 4, device=device)
        try:
            _prototype = nnet.module.prototypes.weight.data.clone()  # [100,1024]
        except:
            _prototype = nnet.prototypes.weight.data.clone()
        print("y", y)
        y = y.repeat_interleave(4)
        print("y_new", y)
        y = _prototype[y].view(_n_samples, -1).to(device)

        print("y.shape", y.shape)
        if config.iter_num < config.iter_num_max:
            kwargs = dict(y=None, is_stage2=False)
        else:
            kwargs = dict(y=y, is_stage2=True)

        _feat = score_model.decode(
            _z_init,
            **kwargs,
        )
        return decode_large_batch(_feat), kwargs

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs, kwargs = sample_fn(img_num)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)  # b c h w

    wandb.log(
        {
            f"eval_cluster_vis_during_training_seed{seed}": [
                wandb.Image(
                    sampled_imgs,
                    caption=f"iter{config.iter_num}/{config.iter_num_max}_is_stage2{int(kwargs['is_stage2'])}_seed{seed}",
                )
            ]
        }
    )


def eval_clusterseg_vis(
    nnet,
    nnet_guidance,
    config,
    accelerator,
    device,
    dataset,
    decode_large_batch,
    sample_path,
    data_generator,
    encode,
    autoencoder,
    img_num=128,
    seed=44,
):
    utils_uvit.set_seed(
        seed,
    )

    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=None,
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
    )
    score_model_guidance = Flow_Matching(net=nnet_guidance)  # maybe buggy here

    def sample_fn(_n_samples):
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        _image, _label = batch
        _n_samples = min(_n_samples, len(_image))

        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if True:
            batch = tree_map(lambda x: x.to(device), next(data_generator))
            _image, _label = batch

            assert len(_image) % 4 == 0

            _image = _image[: len(_image) // 4]
            _image = _image.repeat_interleave(4, dim=0)
            print("_image = _image.repeat_interleave(4, dim=0)")

            x_train = (
                autoencoder.sample(_image)
                if "feature" in config.dataset.name
                else encode(_image)
            )
            sigma_min = config.dynamic.sigma_min
            noise = torch.randn_like(x_train)
            noise = noise[: len(_image) // 4]
            noise = noise.repeat_interleave(4, dim=0)

            ema_t = torch.tensor([config.ema_t] * _n_samples, device=device)
            ema_t_ = ema_t[:, None, None, None]  # [B, 1, 1, 1]
            ema_x_t = ema_t_ * x_train + (1 - (1 - sigma_min) * ema_t_) * noise
            y_cond, dm_emb, _cluster_ids = score_model_guidance.get_cluster_info(
                x_t=ema_x_t,
                t=ema_t,
            )
        else:
            try:
                try:
                    _prototype = score_model_guidance.net.prototypes.weight
                except:
                    _prototype = (
                        score_model_guidance.net.module.prototypes.weight
                    )  # multi-gpu
                K, c_dim = _prototype.shape
                y_cond = torch.FloatTensor(_n_samples).uniform_(0, 1) * K
                y_cond = y_cond.long().to(device)
                # print("y_cond", y_cond)
                y_cond = _prototype[y_cond].view(_n_samples, c_dim).to(device)
                print("y_cond.shape", y_cond.shape)
            except:
                import ipdb

                ipdb.set_trace()

        print("y_cond.shape", y_cond.shape)  # [(b x 256) c]
        # y_cond = y_cond.view(_n_samples, 16, 16, -1)
        kwargs = dict(y=y_cond, is_stage2=True)
        print("_cluster_ids.shape", _cluster_ids.shape)
        # import ipdb

        # ipdb.set_trace()
        origin_images = decode_large_batch(x_train)

        if hasattr(config.sample, "euler_step_size"):
            kwargs.update(step_size=config.sample.euler_step_size)
            _feat = score_model.decode_euler(
                _z_init,
                **kwargs,
            )
        else:
            _feat = score_model.decode(
                _z_init,
                **kwargs,
            )
        return decode_large_batch(_feat), _cluster_ids, origin_images

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs, _cluster_ids, origin_images = sample_fn(img_num)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)  # b c h w
    origin_images = dataset.unpreprocess(origin_images)  # b c h w
    _cluster_ids = _cluster_ids.view(-1, 16, 16)
    _cluster_ids_new = torch.nn.functional.interpolate(
        _cluster_ids.float().unsqueeze(1), scale_factor=16, mode="nearest"
    )
    # _cluster_ids = _cluster_ids.squeeze(1)
    _cluster_ids_new = _cluster_ids_new / 300.0

    wandb.log(
        {
            f"gen_seed{seed}": [wandb.Image(sampled_imgs, caption=f"gen_seed{seed}")],
            f"origin_imgs_seed{seed}": [
                wandb.Image(origin_images, caption=f"origin_imgs_seed{seed}")
            ],
            f"cluster_ids_seed{seed}": [
                wandb.Image(_cluster_ids_new, caption=f"cluster_ids_seed{seed}")
            ],
        }
    )


def vis_cfg_standard(
    nnet, config, accelerator, device, dataset, decode_large_batch, sample_path
):
    def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
        _cond = nnet(x, timesteps, y=y, **kwargs)[0]
        _uncond = nnet(
            x,
            timesteps,
            y=torch.tensor([dataset.K] * x.size(0), device=device),
            **kwargs,
        )[0]
        return _cond + scale * (_cond - _uncond)

    score_model = Flow_Matching(
        net=cfg_nnet,
        rf_kwargs=config.rf_kwargs,
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if config.train.mode == "uncond":
            kwargs = dict()
        elif config.train.mode == "cond":
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        _feat = score_model.decode(
            _z_init,
            **kwargs,
        )
        return decode_large_batch(_feat)

    os.makedirs(sample_path, exist_ok=True)
    sampled_imgs = sample_fn(16)
    sampled_imgs = dataset.unpreprocess(sampled_imgs)
    save_image(sampled_imgs, os.path.join(sample_path, "sampled_imgs.png"))


def evaluate(config):
    print("config.sample.scale", config.sample.scale)
    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # # mp.set_start_method("spawn"), this will cause error in mvl-gpu, comment it, this will cause error in mvl-gpu, comment it
    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    if accelerator.is_main_process:
        if wandb.run is None:
            wandb.login(relogin=True, key=wandb_key)
            wandb.init(project="sgfm_eval", name="sgfm_eval")
            wandb.config.update(config)

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils_uvit.set_logger(log_level="info", fname=config.output_path)
    else:
        utils_uvit.set_logger(log_level="error")
        builtins.print = lambda *args: None

    nnet = utils_uvit.get_nnet(config=config, **config.nnet)
    logging.info(f"load nnet from {config.nnet_path}")
    accelerator.unwrap_model(nnet).load_state_dict(
        torch.load(config.nnet_path, map_location="cpu")
    )
    nnet.eval()

    nnet_guidance = utils_uvit.get_nnet(config=config, **config.nnet)
    nnet_guidance_path = config.nnet_path.replace("nnet.pth", "nnet_ema.pth")
    logging.info(f"load nnet from {nnet_guidance_path}")
    accelerator.unwrap_model(nnet_guidance).load_state_dict(
        torch.load(nnet_guidance_path, map_location="cpu")
    )
    nnet_guidance.eval()

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat), dataset.fid_stat
    train_dataset = dataset.get_split(
        split="train",
        labeled=True,  # alwyas labeled,we need label to calculate NMI
    )
    from torch.utils.data import DataLoader

    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=config.sample.mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    nnet, nnet_guidance, train_dataset_loader = accelerator.prepare(
        nnet,
        nnet_guidance,
        train_dataset_loader,
    )

    def get_data_generator():
        while True:
            for data in tqdm(
                train_dataset_loader,
                disable=not accelerator.is_main_process,
                desc="epoch",
            ):
                yield data

    data_generator = get_data_generator()

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils_uvit.amortize(
            _batch.size(0), decode_mini_batch_size
        ):
            x = decode(_batch[pt : pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    if config.vis_cfg:
        sample_path = "./vis/vis_cfg"
        os.makedirs(sample_path, exist_ok=True)
        vis_cfg(
            nnet=nnet,
            config=config,
            accelerator=accelerator,
            device=device,
            dataset=dataset,
            decode_large_batch=decode_large_batch,
            sample_path=sample_path,
        )
        print("vis cfg done..")
        exit(0)
    elif config.vis_fm_chain:
        sample_path = "./vis_fm_chain"
        os.makedirs(sample_path, exist_ok=True)
        vis_fm_chain(
            nnet=nnet,
            config=config,
            accelerator=accelerator,
            device=device,
            dataset=dataset,
            decode_large_batch=decode_large_batch,
            sample_path=sample_path,
        )
        print("vis fm chain done..")
        exit(0)
    elif config.eval_cluster_vis_during_training:
        print("eval_cluster_vis_during_training start..")
        sample_path = "./eval_cluster_vis_during_training"
        os.makedirs(sample_path, exist_ok=True)
        eval_cluster_vis_during_training(
            nnet=nnet,
            config=config,
            accelerator=accelerator,
            device=device,
            dataset=dataset,
            decode_large_batch=decode_large_batch,
            sample_path=sample_path,
        )
        print("eval_cluster_vis_during_training done..")
        return

    elif config.eval_clusterseg_vis:
        print("eval_clusterseg_vis start..")
        sample_path = "./eval_clusterseg_vis"
        os.makedirs(sample_path, exist_ok=True)
        eval_clusterseg_vis(
            nnet=nnet,
            nnet_guidance=nnet_guidance,
            config=config,
            accelerator=accelerator,
            device=device,
            dataset=dataset,
            decode_large_batch=decode_large_batch,
            sample_path=sample_path,
            data_generator=data_generator,
            encode=encode,
            autoencoder=autoencoder,
        )
        print("eval_clusterseg_vis done..")
        exit(0)

    elif config.vis_cluster_samples_online:
        print("vis_cluster_samples_online start..")
        sample_path = "./vis_cluster_samples_online"
        os.makedirs(sample_path, exist_ok=True)
        vis_cluster_samples_online(
            nnet=nnet,
            config=config,
            accelerator=accelerator,
            device=device,
            dataset=dataset,
            decode_large_batch=decode_large_batch,
            sample_path=sample_path,
        )
        print("vis_cluster_samples_online done..")
        return

    if (
        "cfg" in config.sample and config.sample.cfg and config.sample.scale >= 0
    ):  # classifier free guidance
        logging.info(f"Use classifier free guidance with scale={config.sample.scale}")

        def cfg_nnet(x, timesteps, y, scale=config.sample.scale, **kwargs):
            res = nnet(x, timesteps, y=y, **kwargs)
            _cond = res[0]
            res = nnet(
                x,
                timesteps,
                y=None,
                **kwargs,
            )
            _uncond = res[0]
            return _cond + scale * (_cond - _uncond)

        # set the score_model to train
        print("cfg open..")
        score_model = Flow_Matching(
            net=cfg_nnet,
        )
        score_model_guidance = Flow_Matching(
            net=nnet_guidance,
        )
    else:
        # set the score_model to train
        raise NotImplementedError
        score_model = Flow_Matching(
            net=nnet,
            rf_kwargs=config.rf_kwargs,
        )

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(
        f"sample: n_samples={config.sample.n_samples}, mixed_precision={config.mixed_precision},CFG={config.sample.scale}"
    )

    def sample_fn(_n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if True:
            batch = tree_map(lambda x: x.to(device), next(data_generator))
            _image, _label = batch
            #
            if config.is_debug:
                _n_samples = min(_n_samples, len(_image))
            else:
                assert _n_samples <= len(_image)
            _image = _image[:_n_samples]
            _label = _label[:_n_samples]
            x_train = (
                autoencoder.sample(_image)
                if "feature" in config.dataset.name
                else encode(_image)
            )
            sigma_min = config.dynamic.sigma_min
            noise = torch.randn_like(x_train)
            ema_t = torch.tensor([config.ema_t] * _n_samples, device=device)
            ema_t_ = ema_t[:, None, None, None]  # [B, 1, 1, 1]
            ema_x_t = ema_t_ * x_train + (1 - (1 - sigma_min) * ema_t_) * noise
            y_cond, dm_emb, _cluster_ids = score_model_guidance.get_cluster_info(
                x_t=ema_x_t,
                t=ema_t,
            )
        else:
            try:
                try:
                    _prototype = score_model_guidance.net.prototypes.weight
                except:
                    _prototype = (
                        score_model_guidance.net.module.prototypes.weight
                    )  # multi-gpu
                K, c_dim = _prototype.shape
                y_cond = torch.FloatTensor(_n_samples).uniform_(0, 1) * K
                y_cond = y_cond.long().to(device)
                # print("y_cond", y_cond)
                y_cond = _prototype[y_cond].view(_n_samples, c_dim).to(device)
                print("y_cond.shape", y_cond.shape)
            except:
                import ipdb
                ipdb.set_trace()
        kwargs = dict(y=y_cond, is_stage2=True)

        if hasattr(config.sample, "euler_step_size"):
            kwargs.update(step_size=config.sample.euler_step_size)
            _feat = score_model.decode_euler(
                _z_init,
                **kwargs,
            )
        else:
            _feat = score_model.decode(
                _z_init,
                **kwargs,
            )
        return decode_large_batch(_feat)

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f"Samples are saved in {path}")
        utils_uvit.sample2dir(
            accelerator,
            path,
            config.sample.n_samples,
            config.sample.mini_batch_size,
            sample_fn,
            dataset.unpreprocess,
        )
        if accelerator.is_main_process:
            wand_eval_dict = dict()
            try:
                fid = calculate_fid_given_paths(
                    paths=(dataset.fid_stat, path), num_workers=1
                )
            except Exception as e:
                if config.is_debug:
                    print(e)
                    fid = -1
                else:
                    raise
            logging.info(f"nnet_path={config.nnet_path}, fid={fid}")
            wand_eval_dict["fid"] = fid
            for isc_splits in [1, 10]:
                tf_metrics_dict = torch_fidelity.calculate_metrics(
                    input1=path,
                    cuda=True,
                    isc=True,
                    isc_splits=isc_splits,
                    verbose=False,
                )
                wand_eval_dict[f"is_tf_s{isc_splits}"] = tf_metrics_dict[
                    "inception_score_mean"
                ]

            wand_eval_dict = {
                f"s{config.sample.scale}_{k}": v for k, v in wand_eval_dict.items()
            }
            if hasattr(config.sample, "euler_step_size"):
                wand_eval_dict = {
                    f"stepsize{config.sample.euler_step_size}_{k}": v
                    for k, v in wand_eval_dict.items()
                }
            print(wand_eval_dict)
            wand_eval_dict = {f"eval/{k}": v for k, v in wand_eval_dict.items()}
            wandb.log(wand_eval_dict)


FLAGS = flags.FLAGS
if False:
    config_flags.DEFINE_config_file(
        "config",
        "configs/imagenet100/imagenet100_256_uvit_large_cls_online_ema_sep_v2.py",
        "Training configuration.",
        lock_config=False,
    )
    flags.DEFINE_string(
        "nnet_path",
        "workdir/ema999_AlwaysSwav_sampling0.85_d0_v2.9_uvit_embed_cls_ema_sep_imagenet100_256_features_ema_uncond0.15_ema20000.0_swav1.0_proK300dim1024/ckpts/39920.ckpt/nnet.pth",
        "The nnet to evaluate.",
    )
    flags.DEFINE_string("output_path", "output.log", "The path to output log.")
elif True:  #
    config_flags.DEFINE_config_file(
        "config",
        "configs/imagenet256/imagenet256_uvit_large_cls_online_ema_sep_v2.py",
        "Training configuration.",
        lock_config=False,
    )
    flags.DEFINE_string(
        "nnet_path",
        "workdir/in256_sg_ema0.5_d0_v2.9_uvit_embed_cls_ema_sep_imagenet256_features_ema_uncond0.15_ema150000.0_swav1.0_proK300dim1024/ckpts/39920.ckpts/nnet.pth",
        "The nnet to evaluate.",
    )
    flags.DEFINE_string("output_path", "output.log", "The path to output log.")

elif False:
    config_flags.DEFINE_config_file(
        "config",
        "configs/imagenet100/imagenet100_256_uvit_large_cls_online_ema_patch.py",
        "Training configuration.",
        lock_config=False,
    )
    flags.DEFINE_string(
        "nnet_path",
        "workdir/seg_bs512_d0_v2.8_uvit_embed_cls_ema_patch_imagenet100_256_features_ema_uncond0.15_ema30000.0_swav1.0_proK300dim1024/ckpts/60000.ckpt/nnet.pth",
        "The nnet to evaluate.",
    )
    flags.DEFINE_string("output_path", "output.log", "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path

    # config.sample.scale = FLAGS.sample.scale

    config.sample.scalelist = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        2.0,
        2.5,
        3,
        3.5,
        4,
        4.5,
        5,
        5.5,
        6,
        6.5,
        7,
        7.5,
        8,
    ]

    config.sample.steplist = [1, 10, 20, 50, 100, 200, 500, 1000]

    if config.is_debug:
        config.sample.n_samples = 100
        config.sample.mini_batch_size = 20
        config.sample.scale = 10.0
    if config.eval_scalelist and hasattr(config.sample, "scalelist"):
        print("scale is list, evaluate multi-scale")
        for scale in config.sample.scalelist:
            config.sample.scale = float(scale)
            evaluate(config)
    elif config.eval_steplist and hasattr(config.sample, "steplist"):
        print("step is list, evaluate multi-step")
        for step in config.sample.steplist:
            print("*" * 88)
            print("euler_step_size=", float(1.0 / step))
            config.sample.euler_step_size = float(1.0 / step)
            evaluate(config)
    elif config.eval_cluster_vis_during_training:
        root_path = config.nnet_path
        if root_path.endswith("/nnet.pth"):
            root_path = root_path.replace("/39920.ckpts/nnet.pth", "")
        print("root_path", root_path)
        ckpt_list = os.listdir(root_path)

        ckpt_list = sorted(ckpt_list, key=lambda x: int(x.split(".")[0]))
        print("ckpt_list", ckpt_list)
        config.iter_num_max = int(ckpt_list[-1].split(".")[0])
        print("config.iter_num_max", config.iter_num_max)
        for iii in ckpt_list:
            iter_num = int(iii.split(".")[0])
            config.iter_num = iter_num

            config.nnet_path = os.path.join(root_path, iii) + "/nnet.pth"
            evaluate(config)

    else:
        print("scaleee is single, evaluate single-scale, scale=", config.sample.scale)
        evaluate(config)


if __name__ == "__main__":
    app.run(main)

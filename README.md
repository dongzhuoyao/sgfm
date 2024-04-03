# Guided Diffusion from Self-Supervised Diffusion Features




# Dataset Preparation

## Misc
```
mkdir assets && cd assets
prepare assets/fid_stats/ 
assets/stable-diffusion/ 
assets/pretrained_weights/ 
assets/datasets/imagenet256_features.tar.gz 
```

```
python scripts/extract_imagenet_feature.py ~/data/imagenet
```


## LSUN-churches
```
download lsun-churches, git clone lsun repo
python3 download.py -c church_outdoor
and unzip them,unzip church_outdoor_train_lmdb.zip
python data.py export church_outdoor_train_lmdb --out_dir churches_flat_train --flat
```

# Environment Preparation


we use python==3.10 and torch==2.0


```bash 
conda create -n sgfm  python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections
pip install hydra-core wandb einops scikit-learn --upgrade
pip install einops scikit-learn  webdataset
pip install transformers==4.23.1 pycocotools # for text-to-image task
pip install ml_collections einops h5py
pip install accelerate==0.23.0
pip install git+https://github.com/dongzhuoyao/pytorch-fid-with-sfid
pip install torch_fidelity  clean_fid fire
pip install python-dotenv #for configuring the wandb key
```

faiss install 
```bash 
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```


# Main Experiment

ImageNet100, Image-level
```bash
CUDA_VISIBLE_DEVICES=2  accelerate launch  --num_processes 1 --main_process_port 8050  --mixed_precision fp16 train_sgfm_hydra.py nnet=uvit_online  train.batch_size=64  train.n_steps=400_000 train.log_interval=10 train.vis_interval=5_000  train.save_interval=40_000  is_debug=0 tag=_
```

ImageNet100, Patch-level
```bash
CUDA_VISIBLE_DEVICES=2  accelerate launch  --num_processes 1 --main_process_port 8050  --mixed_precision fp16 train_sgfm_hydra.py nnet=uvit_online_patch train.batch_size=64  train.n_steps=400_000 train.log_interval=10 train.vis_interval=5_000  train.save_interval=40_000  is_debug=0 tag=_
```


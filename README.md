# DiffuSETS

Note: All the python commands below are run at current directory i.e. `/.../text2ecg/`

## Inference and evaluation

To run inference:

```python
python DiffuSETS_inference.py
```

For evaluation, see `metrics.ipynb`

## Training

vanilla training script is `DiffuSETS_train.py`, can be launched through 

```sh
python DiffuSETS_train.py
```

For further experiment such as ablation study, it is highly recommended to create a new folder to contain new training scripts

For example, create directories of `./ablation/hr/` with `train.py`, then it can be launched by 

```sh
python -m ablation.hr.train 
```

## Repository structure

### `checkpoints/` : experiment result container

### `dataset/` : dataset module 

Include mimic_iv_ecg module and ptbxl module as well as related VAE latent dataset module. 

### `unet/` : unet models

For new unet, simply add to this folder and use following expressions to have it included in other scripts

```python
from unet.new_added_unet import xxx
```

### `vae/` : scripts related to vae model

To encode ECG into latent:

```sh
python -m vae.vae_encoding
```

To launch a vae model training:

```sh
python -m vae.vae_train
```

To evaluate vae performance:

```sh
python -m vae.vae_test
```

### `clip/` : scripts related to clip model

To launch a clip model training: 

```sh
python -m clip.clip_train
```

For evaluation, see `clip_test.ipynb`
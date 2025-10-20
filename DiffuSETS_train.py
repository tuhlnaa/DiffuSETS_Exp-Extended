import argparse
import json
import logging
import os
import torch
import wandb

from diffusers import DDPMScheduler
from pathlib import Path
from torch.utils.data import DataLoader

# Import custom modules
from dataset.mimic_iv_ecg_datasetV2 import MIMIC_IV_ECG_VAE_Dataset
from utils.config import init_seeds 


def parse_arg():
    parser = argparse.ArgumentParser(description='DiffuSETS Training') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    init_seeds()
    with open(args.config, 'r') as f:
        config = json.load(f)

    wandb.init(project="DiffuSETS", group="Training", entity=None)
    wandb.run.name = "Experiment 1"

    meta = config['meta']
    roots = config['dependencies']
    hyperparams = config['hyper_para']

    k_max = 0
    for item in os.listdir(roots['checkpoints_dir']):
        if meta['exp_type'] + "_" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = os.path.join(roots['checkpoints_dir'], f"{meta['exp_type']}_{k_max + 1}")

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f"{meta['exp_type']}_{k_max + 1}")
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(meta)
    logger.info(hyperparams)

    # train_dataset = DictDataset(roots['dataset_path']) 
    # train_dataloader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'])

    # Load dataset
    dataset_path = Path(roots['dataset_path'])
    #train_dataset = DictDataset(dataset_path)
    train_dataset = MIMIC_IV_ECG_VAE_Dataset(dataset_path, subset_proportion=0.005)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=hyperparams.get('num_workers', 0)
    )
    

    use_vae_latent = meta['vae_latent']     
    n_channels = 4 if use_vae_latent else 12 

    if meta['condition']:
        from unet.unet_conditional import ECGConditional 

        unet = ECGConditional(hyperparams['num_train_steps'], kernel_size=hyperparams['unet_kernel_size'], num_levels=hyperparams['unet_num_level'], n_channels=n_channels)
    else: 
        from unet.unet_nocondition import ECGNoCondition 

        unet = ECGNoCondition(hyperparams['num_train_steps'], kernel_size=hyperparams['unet_kernel_size'], num_levels=hyperparams['unet_num_level'], n_channels=n_channels)

    diffused_model = DDPMScheduler(num_train_timesteps=hyperparams['num_train_steps'], beta_start=hyperparams['beta_start'], beta_end=hyperparams['beta_end'])

    if use_vae_latent:
        from utils.train import train_model 

        train_model(
            meta=meta, 
            save_weights_path=save_weights_path, 
            dataloader=train_dataloader, 
            diffused_model=diffused_model, 
            unet=unet, 
            hyperparams=hyperparams, 
            logger=logger
        )

    else:
        from vae.vae_model import VAEDecoder 
        from utils.train_novae import train_model_novae

        decoder = VAEDecoder() 
        checkpoint = torch.load(roots['vae_path'], map_location='cpu')
        decoder.load_state_dict(checkpoint['decoder'])

        train_model_novae(
            meta=meta, 
            save_weights_path=save_weights_path, 
            dataloader=train_dataloader, 
            diffused_model=diffused_model, 
            unet=unet, 
            decoder=decoder, 
            hyperparams=hyperparams, 
            logger=logger
        )


if __name__ == '__main__': 
    main()

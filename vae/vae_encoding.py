import os
import torch
from torch.utils.data import Dataset
from vae.vae_model import VAE_Encoder

from dataset.mimic_iv_ecg_dataset import MIMIC_IV_ECG_Dataset
from tqdm import tqdm

@torch.no_grad()
def encode_dataset_to_latent(dataset: Dataset, 
                             vae_encoder: VAE_Encoder, 
                             target_path: str, 
                             device: str):
    try:
        os.makedirs(target_path)
    except:
        pass

    encoder.to(device)
    for idx, (X, y) in enumerate(tqdm(dataset)):
        # X: (L, C) -> (1, L, C)
        X = X.unsqueeze(0)
        X = X.to(device)

        # X: (1, L, C) -> latent: (4, L / 8)
        latent, _, __ = vae_encoder(X)
        latent = latent.squeeze(0)

        save_dict = {
            'data': latent.cpu(), 
            'label': y
        }

        torch.save(save_dict, os.path.join(target_path, f'{idx}.pt'))

if __name__ == '__main__':
    device = 'cuda:1'
    target_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae_backup'

    path = '/data1_science/1shared/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    dataset = MIMIC_IV_ECG_Dataset(path, usage='all', resample_length=1024)

    vae_path = './checkpoints/vae_1/VAE_model_ep9.pth'
    vae_weight_dict = torch.load(vae_path, map_location=device) 
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_weight_dict['encoder'])

    encode_dataset_to_latent(dataset=dataset, 
                             vae_encoder=encoder, 
                             target_path=target_path, 
                             device=device)


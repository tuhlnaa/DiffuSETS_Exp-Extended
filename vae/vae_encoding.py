import os
import torch
import pandas as pd
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
    save_dict = dict()

    patient_table_path = '/data/0shared/MIMIC/mimic-iv-2.2/hosp/patients.csv'
    patient_table = pd.read_csv(patient_table_path, index_col='subject_id', low_memory=False)

    exclude_list = []
    for idx, (X, label) in enumerate(tqdm(dataset)):
        # X: (L, C) -> (1, L, C)
        X = X.unsqueeze(0)
        X = X.to(device)

        # X: (1, L, C) -> latent: (4, L / 8)
        latent, _, __ = vae_encoder(X)
        latent = latent.squeeze(0)

        if label['hr'] > 99998:
            exclude_list.append(idx)
            continue

        patient_id = label['subject_id']
        query = patient_table[patient_table.index == patient_id] 

        # mimic iv note do not release
        # note_id = label['note_id']
        # label['text'] = note_table[note_table['note_id'] == note_id]['text'].to_list()[0]

        try:
            label['age'] = query['anchor_age'].to_list()[0]
            label['gender'] = query['gender'].to_list()[0]

        except IndexError:
            exclude_list.append(idx)
            continue

        save_dict[idx] = {
            'data': latent.cpu(), 
            'label': label
        }

        if idx == 49999:
            torch.save(save_dict, os.path.join(target_path, 'mimic_vae_lite.pt'))

    torch.save(save_dict, os.path.join(target_path, 'mimic_vae.pt'))
    print(len(exclude_list))

if __name__ == '__main__':
    device = 'cuda:0'
    target_path = '/data/0shared/laiyongfan/data_text2ecg/'

    path = '/data/0shared/MIMIC/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    dataset = MIMIC_IV_ECG_Dataset(path, usage='all', resample_length=1024)

    vae_path = '/data/0shared/laiyongfan/data_text2ecg/models/vae_model.pth'
    vae_weight_dict = torch.load(vae_path, map_location=device) 
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_weight_dict['encoder'])

    encode_dataset_to_latent(dataset=dataset, 
                             vae_encoder=encoder, 
                             target_path=target_path, 
                             device=device)


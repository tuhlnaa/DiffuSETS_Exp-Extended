import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler
from unet.conditional_unet_patient_3 import ECGconditional
from dataset.ptbxl_dataset import PtbxlDataset_VAE
from vae.vae_model import VAE_Decoder

import time
import sys
import os
import pandas as pd
import numpy as np
import logging

embedding_dict_ptbxl = pd.read_csv('/data/0shared/chenjiabo/DiffuSETS/data/ptbxl_database_embed.csv', low_memory=False)[['ecg_id', 'text_embed']]
original_sheet = pd.read_csv('/data/0shared/laiyongfan/data_text2ecg/ptb-xl/ptbxl_database.csv', low_memory=False)
embedding_dict_ptbxl = pd.merge(embedding_dict_ptbxl, original_sheet)

def fetch_text_embedding_ptbxl(text:str):
    text = text.split('|')[0]
    text = text.replace('The report of the ECG is that ', '')
    try:
        text_embed = embedding_dict_ptbxl.loc[embedding_dict_ptbxl['report'] == text, 'text_embed'].values[0]
        text_embed = eval(text_embed)
    except IndexError:
        text_embed = [0] * 1536
    # print(text_embed)
    return torch.tensor(text_embed)

feature_vec = {
    'text': 1,
    'gender': 1,
    'age': 1,
    'heart rate': 1}

def train_epoch_channels(dataloader, 
                         net: ECGconditional, 
                         diffused_model: DDPMScheduler, 
                         optimizer, 
                         device, 
                         decoder, 
                         number_of_repetition=1):
    loss_list = []
    net.train()
    for _ in range(number_of_repetition):
        for data, label in dataloader:
            #batch_size random int variables from 1 to Tmax-1
            # t: (batch_size, )
            # data = data.to(device)
            # label = label.to(device)
            gender = []
            age = label['age']
            hr = label['hr']

            condition = {}
            
            if feature_vec['gender']:
                for ch in label['gender']:
                    if ch == 'M':
                        gender.append(1)
                    else:
                        gender.append(0)
                gender = np.array(gender)
                gender = np.repeat(gender[:, np.newaxis], 1, axis=1)
                gender = np.repeat(gender[:, :, np.newaxis], 1, axis=2)
                gender = torch.Tensor(gender)
                gender = gender.to(device)
                condition.update({'gender': gender})

            if feature_vec['age']:
                age = np.array(age)
                age = np.repeat(age[:, np.newaxis], 1, axis=1)
                age = np.repeat(age[:, :, np.newaxis], 1, axis=2)
                age = torch.Tensor(age)
                age = age.to(device)
                condition.update({'age': age})

            if feature_vec['heart rate']:
                hr = np.array(hr)
                hr = np.repeat(hr[:, np.newaxis], 1, axis=1)
                hr = np.repeat(hr[:, :, np.newaxis], 1, axis=2)
                hr = torch.Tensor(hr)
                hr = hr.to(device)
                condition.update({'heart rate': hr})

            texts = label['text']
            texts = [fetch_text_embedding_ptbxl(x) for x in texts]
            # (B, 1535) -> (B, 1, 1536)
            text_embed = torch.stack(texts).unsqueeze(1)

            data = data.to(device)
            # (B, L, C) -> (B, C, L)
            ecg = decoder(data).transpose(-1, -2)
            text_embed = text_embed.to(device)

            t = torch.randperm(diffused_model.config.num_train_timesteps-2)[:ecg.shape[0]] + 1 

            noise = torch.randn(ecg.shape, device=ecg.device)
            xt = diffused_model.add_noise(ecg, noise, t)

            xt = xt.to(device)
            t = t.to(device)
            noise = noise.to(device)

            
            for key in condition:
                condition[key] = condition[key].to(device)
            # condition.to(device)

            # change this line to fit your Unet
            nosie_estim = net(xt, t, text_embed, condition)

            # Batchwise MSE loss 
            loss = F.mse_loss(nosie_estim, noise, reduction='sum').div(noise.size(0))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return sum(loss_list) / len(loss_list)


if __name__ == "__main__":

    k_max = 0
    for item in os.listdir('./checkpoints'):
        if "unet_" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/unet_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'unet{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    H_ = {
        'lr': 1e-5, 
        'batch_size': 64, 
        'epochs': 200, 
    }
    logger.info(H_)

    PTB_VAE_PATH = '/data/0shared/laiyongfan/data_text2ecg/ptb-xl_vae'
    n_channels = 12
    num_train_steps = 1000
    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    
    dataset = PtbxlDataset_VAE(path=PTB_VAE_PATH)
    
    net = ECGconditional(num_train_steps, kernel_size=7, num_levels=5, n_channels=n_channels)
    
    device_str = "cuda:6"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    dataloader = DataLoader(dataset, batch_size=H_['batch_size'])
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=H_['lr'])
    min_loss = 1000

    decoder = VAE_Decoder()
    vae_path = '/data/0shared/laiyongfan/data_text2ecg/models/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    start_time = time.time()

    for i in range(1, H_['epochs'] + 1):
        s_t = time.time()
        mean_loss = train_epoch_channels(dataloader, net, diffused_model, optimizer, device, decoder, number_of_repetition=1)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(net.state_dict(), os.path.join(save_weights_path, 'unet_best.pth'))
            logger.info(f'epoch {i} unet_best.pth has been saved.')
        if (i % 50 == 0):
            torch.save(net.state_dict(), os.path.join(save_weights_path, f'unet_{i}.pth'))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")


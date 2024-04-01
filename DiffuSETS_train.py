import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler
from unet.conditional_unet_patient_3 import ECGconditional
from dataset.mimic_iv_ecg_dataset import DictDataset

import time
import sys
import os
import pandas as pd
import numpy as np
import logging

text_emb = pd.read_csv('./mimic_iv_text_embed.csv')

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
                         number_of_repetition=1):
    loss_list = []
    net.train()
    for _ in range(number_of_repetition):
        for data, label in dataloader:
            #batch_size random int variables from 1 to Tmax-1
            # t: (batch_size, )
            # data = data.to(device)
            # label = label.to(device)
            ecg = data
            label['text']
            label['gender']
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

            text_embed = []
            if feature_vec['text']:
                for text in label['text']:
                    input = ''
                    for ch in text:
                        if ch == '|':
                            break
                        input += ch
                    # print(input)
                    if len(input) >= 1 and input[-1] != '.':
                        input += '.'
                    if len(text_emb.loc[text_emb['text'] == input, 'embed']) > 0:
                        emb = text_emb.loc[text_emb['text'] == input, 'embed'].values[0]
                    else:
                        emb = text_emb.iloc[-1]['embed']
                    # print(emb)
                    emb = eval(emb)
                    # emb = np.repeat(text_embed[:, np.newaxis], 1, axis=1)
                    # emb = torch.Tensor(emb)
                    text_embed.append(emb)

                text_embed = np.array(text_embed)
                text_embed = np.repeat(text_embed[:, np.newaxis, :], 1, axis=1)
            
                text_embed = torch.Tensor(text_embed)

            ecg = ecg.to(device)
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
        'batch_size': 512, 
        'epochs': 200, 
    }
    logger.info(H_)

    # vae_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae_lite.pt'
    vae_path = 'mimic_vae.pt'
    n_channels = 4
    num_train_steps = 1000
    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    
    logger.info('Loading dataset...')
    dataset = DictDataset(path=vae_path)
    logger.info('Done!')
    
    net = ECGconditional(num_train_steps, kernel_size=7, num_levels=5, n_channels=n_channels)
    
    device_str = "cuda:3"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    dataloader = DataLoader(dataset, batch_size=H_['batch_size'])
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=H_['lr'])
    min_loss = 50

    start_time = time.time()

    for i in range(1, H_['epochs'] + 1):
        s_t = time.time()
        mean_loss = train_epoch_channels(dataloader, net, diffused_model, optimizer, device, number_of_repetition=1)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(net.state_dict(), os.path.join(save_weights_path, 'unet_best.pth'))
            logger.info(f'epoch {i} unet_best.pth has been saved.')
        if (i % 50 == 0):
            torch.save(net.state_dict(), os.path.join(save_weights_path, f'unet_{i}.pth'))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")


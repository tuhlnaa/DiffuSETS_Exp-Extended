import torch
from vae.vae_model import VAE_Decoder
from matplotlib import pyplot as plt
import numpy as np 
import wfdb
from tqdm import tqdm
import pandas as pd
import json
import math
from dataset.mimic_iv_ecg_dataset import DictDataset
from unet.unet_conditional import ECGconditional
from unet.unet_nocondition import ECGnocondition 
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os

    
def generation_from_net(diffused_model: DDPMScheduler, net: ECGconditional, batch_size, device, text_embed, condition, use_vae_latent):
    net.eval()
    n_channels = 4 if use_vae_latent else 12 
    dim = 128 if use_vae_latent else 1024
    xi = torch.randn(batch_size, n_channels, dim)
    xi = xi.to(device)
    timesteps = tqdm(diffused_model.timesteps)
    for _, i in enumerate(timesteps):
        t = i*torch.ones(batch_size, dtype=torch.long)
        with torch.no_grad():

            # change this line to fit your unet 
            if condition: 
                noise_predict = net(xi, t, text_embed, condition)
            else: 
                noise_predict = net(xi, t, text_embed)

            xi = diffused_model.step(model_output=noise_predict, 
                                     timestep=i, 
                                     sample=xi)['prev_sample']
    return xi 

def batch_generate_ECG(nums, 
                       batch, 
                       save_path, 
                       exp_type, 
                       test_dataloader, 
                       net, 
                       diffused_model, 
                       device,  
                       use_condition, 
                       use_vae_latent, 
                       verbose=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert exp_type in ['normal', 'af', 'pvc']
    index = 0
    result_array = []  
    # embedding_dict_mimic= pd.read_csv('./mimic_iv_text_embed.csv')
    for _, (x, y) in enumerate(test_dataloader):
        # input_ = x.squeeze(0).detach().numpy()
        # encoder_noise = torch.randn(latent_shape)
        # latent, mu, log_var = encoder(x)
        if index == nums:
            break

        latent = x
        text = y['text'][0].lower()
        gender = 1 if y['gender'] == 'M' else 0
        gender = torch.tensor([gender])
        age = y['age']
        hr = y['hr']

        # if not ('pvc' in text or 'ventricular premature' in text or 'premature ventricular' in text):
        #     continue 
        # if not ('atrial fibrillation') in text:
        #     continue
        if 'normal ecg' in text:
            continue

        print(text)
        index += 1

        text_embed = y['text_embed']
        text_embed = np.array(text_embed)
        text_embed = np.repeat(text_embed[np.newaxis, :], 1, axis=0)
        text_embed = np.repeat(text_embed[np.newaxis, :, :], batch, axis=0)

        text_embed = torch.Tensor(text_embed).squeeze(-1)
        text_embed = text_embed.to(device)
        if verbose:
            print(text_embed.shape)

        if use_condition: 
            condition = {'gender': gender, 'age': age, 'heart rate': hr}

            for key in condition:
                condition[key] = np.array([condition[key]])
                condition[key] = np.repeat(condition[key][np.newaxis, :], batch, axis=0)
                if verbose:
                    print(condition[key].shape)
                condition[key] = torch.Tensor(condition[key])
                condition[key] = condition[key].to(device)

            if verbose:
                print(condition)

        else:
            condition=None

        latent = generation_from_net(diffused_model, net, batch_size=batch, device=device, text_embed=text_embed, condition=condition, use_vae_latent=use_vae_latent)

        result_array.append(latent.detach().cpu().numpy()) 
    
    result_array = np.concatenate(result_array, axis=0)
    print(result_array.shape)
    np.save(os.path.join(save_path, f'completion_{exp_type}.npy'), result_array)

if __name__ == "__main__":
    nums = 20 # 选用的mimic样本数
    batch = 512 #使用[每个样本对应的condition]生成的ECG个数
    use_vae_latent = True 
    use_condition = True 
    exp_type = 'normal'

    save_path = './prerequisites/clf_data'
    device_str = "cuda:2"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    mimic_path = './prerequisites/mimic_vae_lite_0_new.pt'
    mimic_test_data = DictDataset(path=mimic_path)
    mimic_test_dataloader = DataLoader(mimic_test_data, batch_size=1, shuffle=True)

    n_channels = 4 if use_vae_latent else 12 
    num_train_steps = 1000
    net = ECGconditional(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)

    unet_path = './prerequisites/unet_all.pth'
    net.load_state_dict(torch.load(unet_path, map_location=device))
    net = net.to(device)

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = './prerequisites/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    batch_generate_ECG(nums=nums, 
                       batch=batch, 
                       save_path=save_path, 
                       exp_type=exp_type,
                       test_dataloader=mimic_test_dataloader, 
                       net=net, 
                       diffused_model=diffused_model, 
                       device=device,
                       use_condition=use_condition, 
                       use_vae_latent=use_vae_latent)

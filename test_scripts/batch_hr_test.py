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
from wfdb import processing


def find_power_of_ten(number):
    if number > 0:
        power = math.log(number, 10)
        return math.ceil(power)
    else:
        return "Number must be greater than 0"
    
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

def detect_hr(ecg):
    fs = 102.4
    for lead in range(12):
        xqrs = processing.XQRS(sig=ecg[:, lead], fs=fs)
        xqrs.detect(verbose=False)
        qrs_inds = xqrs.qrs_inds
        if len(qrs_inds) > 1:
            rr_intervals = np.diff(qrs_inds) / fs
            heart_rate = 60 / np.mean(rr_intervals)
            break
    
    return heart_rate

def batch_hr_test(nums, 
                batch, 
                save_path, 
                test_dataloader, 
                net, 
                diffused_model, 
                decoder, 
                device,  
                use_condition, 
                use_vae_latent, 
                verbose=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    text_visited = []
    index = 0
    # embedding_dict_mimic= pd.read_csv('./prerequisites/mimic_iv_text_embed.csv')
    loss = 0
    scatters = []
    for _, (x, y) in enumerate(test_dataloader):
        # input_ = x.squeeze(0).detach().numpy()
        # encoder_noise = torch.randn(latent_shape)
        # latent, mu, log_var = encoder(x)
        if index == nums:
            break

        latent = x
        text = y['text'][0]
        gender = 1 if y['gender'] == 'M' else 0
        gender = torch.tensor([gender])
        age = y['age']
        hr = y['hr']

        # text = text.split('|')[0]

        # if len(text) > 0 and text[-1] != '.':
        #     text += '.'

        # if text in text_visited:
        #     continue
        # index += 1

        # print('Diagnosis: The report of the ECG is that {' + text + '}.')
        # text_visited.append(text)

        # try:
        #     text_embed = embedding_dict_mimic.loc[embedding_dict_mimic['text'] == text, 'embed'].values[0]
        #     text_embed = eval(text_embed)
        # except IndexError:
        #     if verbose:
        #         print('text_embedding missing Encoutered')
        #     text_embed = embedding_dict_mimic.iloc[-1]['embed']
        #     text_embed = eval(text_embed)

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

        if use_vae_latent: 
            gen_ecg = decoder(latent)
        else: 
            gen_ecg = latent.transpose(-1, -2) 

        hr_list = []
        for j in range(batch):
            output = gen_ecg[j]

            output_ = output.squeeze(0).detach().cpu().numpy()
            test_hr = detect_hr(output_)
            hr_list.append(test_hr)

        hr = hr.item()
        scatters.append([hr_list[1], hr])
        hr_list = np.array(hr_list)
        loss += np.abs(hr_list - hr).mean() 

        index += 1

    loss /= nums
    print(loss) 
    # np.save(os.path.join(save_path, 'scatters_all.npy'), scatters)



if __name__ == "__main__":
    nums = 10  
    batch = 5 
    use_vae_latent = True 
    use_condition = True 
    save_img = False
    save_path = './exp'
    unet_path = './prerequisites/unet_all.pth'
    device_str = "cuda:1"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    mimic_path = './prerequisites/mimic_vae_lite_0_new.pt'
    mimic_test_data = DictDataset(path=mimic_path)
    mimic_test_dataloader = DataLoader(mimic_test_data, batch_size=1, shuffle=True)

    n_channels = 4 if use_vae_latent else 12 
    num_train_steps = 1000
    if use_condition: 
        net = ECGconditional(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)
    else:
        net = ECGnocondition(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)

    net.load_state_dict(torch.load(unet_path, map_location=device))
    net = net.to(device)

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = './prerequisites/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    batch_hr_test(nums=nums, 
                    batch=batch, 
                    save_path=save_path, 
                    test_dataloader=mimic_test_dataloader, 
                    net=net, 
                    diffused_model=diffused_model, 
                    decoder=decoder, 
                    device=device,
                    use_condition=use_condition, 
                    use_vae_latent=use_vae_latent)

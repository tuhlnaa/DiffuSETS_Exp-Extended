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
import argparse


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
    timesteps = diffused_model.timesteps
    for _, i in enumerate(timesteps):
        t = i*torch.ones(batch_size, dtype=torch.long)
        with torch.no_grad():

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
                       disease,
                       test_dataloader, 
                       net, 
                       diffused_model, 
                       decoder, 
                       device,  
                       use_condition, 
                       use_vae_latent, 
                       save_img=False, 
                       verbose=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    index = 0
    for _, (x, y) in enumerate(test_dataloader):
        if index == nums:
            break

        latent = x
        text = y['text'][0].lower()
        gender = 1 if y['gender'] == 'M' else 0
        gender = torch.tensor([gender])
        age = y['age']
        hr = y['hr']
        subject_id = y['subject_id']
        features_file_content = {}

        if 'warning:' in text:
            continue
        if disease == 'pac' and not ('atrial premature contraction' in text or 'pac(s)' in text): 
            continue
        if disease == 'pvc' and not ('pvc' in text or 'ventricular premature' in text or 'premature ventricular' in text):
            continue 
        if disease == 'lbbb' and not ('left bundle branch block' in text):
            continue
        if disease == 'rbbb' and not ('right bundle branch block' in text):
            continue
        if disease == 'sn' and not ('sinus rhythm' in text):
            continue
        if disease == 'snt' and not ('sinus tachycardia' in text):
            continue
        if disease == 'snb' and not ('sinus bradycardia' in text):
            continue
        if disease == 'sna' and not ('sinus arrhythmia' in text):
            continue
        if disease == 'afl' and not ('atrial flutter' in text):
            continue
        if disease == 'af' and not ('atrial fibrillation' in text):
            continue
        if disease == 'pacing' and not ('pacing' in text or 'pace' in text):
            continue
        if disease == 'mi' and not ('infarct' in text):
            continue
        if disease == 'st' and not ('st junctional' in text or ' st ' in text or 'st-' in text):
            continue
        if disease == 'avbi' and not ('degree' in text):
            continue 
        if disease == 'normal' and ('normal ecg' not in text or 'abnormal ecg' in text):
            continue


        index += 1

        text_embed = y['text_embed']

        features_file_content.update({"batch": batch}) # batch_size
        features_file_content.update({"subject_id": subject_id.item()}) 
        features_file_content.update({"Diagnosis": text}) # prompts

        text_embed = np.array(text_embed)
        text_embed = np.repeat(text_embed[np.newaxis, :], 1, axis=0)
        text_embed = np.repeat(text_embed[np.newaxis, :, :], batch, axis=0)

        text_embed = torch.Tensor(text_embed).squeeze(-1)
        features_file_content.update({"text_embed": str(text_embed.tolist())}) 
        text_embed = text_embed.to(device)
        if verbose:
            print(text_embed.shape)

        if use_condition: 
            condition = {'gender': gender, 'age': age, 'heart rate': hr}

            for key in condition:
                features_file_content.update({key: condition[key].item()}) 

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

        features_file_content.update({"Ori Latent": str(np.array(latent).tolist())}) 
        latent = generation_from_net(diffused_model, net, batch_size=batch, device=device, text_embed=text_embed, condition=condition, use_vae_latent=use_vae_latent)
        features_file_content.update({"Gen Latent": str(np.array(latent.cpu()).tolist())}) 

        number_str = str(index).zfill(find_power_of_ten(nums))
        save_sample_path = os.path.join(save_path, number_str)
        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)

        if save_img:
            input_ = decoder(x.to(device))
            input_ = input_.squeeze(0).detach().cpu().numpy()
            wfdb.plot_items(input_, figsize=(10, 10), title="Original ECG") 
            plt.savefig(os.path.join(save_sample_path, 'Original ECG.png'))
            plt.close()

            if use_vae_latent: 
                gen_ecg = decoder(latent)
            else: 
                gen_ecg = latent.transpose(-1, -2) 

            for j in range(batch):
                output = gen_ecg[j]

                output_ = output.squeeze(0).detach().cpu().numpy()
                wfdb.plot_items(output_, figsize=(10, 10), title="Generated ECG")
                plt.savefig(os.path.join(save_sample_path, f'{j} Generated ECG.png'))
                plt.close()

        with open(os.path.join(save_sample_path, 'features.json'), 'w') as json_file:
            json.dump(features_file_content, json_file, indent=4)
            # print(f"Features has been successfully written to {save_sample_path}features.json")
    
    print(f'{exp_type}: {index} x {batch} done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple way to manage experiments"
    )

    # Add arguments
    parser.add_argument(
        "--exp_type", type=str, required=True,
        help="experimetn type in ['pvc', 'pac', ...]"
    )
    parser.add_argument(
        "--gpu_ids", type=int, default=0,
        help="gpu index"
    )
    parser.add_argument(
        "--nums", type=int, default=50,
        help="num of generations"
    )
    parser.add_argument(
        "--batch", type=int, default=10,
        help="num of ecg in one generation"
    )
    args = parser.parse_args()
    nums = args.nums 
    batch = args.batch 
    exp_type = args.exp_type

    save_path = f'./exp/disease/{exp_type}'
    use_vae_latent = True 
    use_condition = True 

    save_img = False
    device_str = f"cuda:{args.gpu_ids}"
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
                       disease=exp_type,
                       test_dataloader=mimic_test_dataloader, 
                       net=net, 
                       diffused_model=diffused_model, 
                       decoder=decoder, 
                       device=device,
                       use_condition=use_condition, 
                       use_vae_latent=use_vae_latent, 
                       save_img=save_img)


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
from unet.conditional_unet_patient_3 import ECGconditional
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os


def find_power_of_ten(number):
    if number > 0:
        power = math.log(number, 10)
        return math.ceil(power)
    else:
        return "Number must be greater than 0"
    
def generation_from_net(diffused_model: DDPMScheduler, net: ECGconditional, batch_size, device, text_embed, condition, dim=128):
    net.eval()
    xi = torch.randn(batch_size, 4, dim)
    xi = xi.to(device)
    timesteps = tqdm(diffused_model.timesteps)
    for _, i in enumerate(timesteps):
        t = i*torch.ones(batch_size, dtype=torch.long)
        with torch.no_grad():

            # change this line to fit your unet 
            noise_predict = net(xi, t, text_embed, condition)

            xi = diffused_model.step(model_output=noise_predict, 
                                     timestep=i, 
                                     sample=xi)['prev_sample']
    return xi 

def batch_generate_ECG(nums, 
                       batch, 
                       save_path, 
                       test_dataloader, 
                       net, 
                       diffused_model, 
                       decoder, 
                       device,  
                       save_img=False, 
                       verbose=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not save_img:
        print("Ignore image drawing and saving...")

    text_visited = []
    index = 0
    embedding_dict_mimic= pd.read_csv('./mimic_iv_text_embed.csv')
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
        subject_id = y['subject_id']

        features_file_content = {}

        text = text.split('|')[0]

        if len(text) > 0 and text[-1] != '.':
            text += '.'

        if text in text_visited:
            continue
        index += 1

        print('Diagnosis: The report of the ECG is that {' + text + '}.')
        text_visited.append(text)

        try:
            text_embed = embedding_dict_mimic.loc[embedding_dict_mimic['text'] == text, 'embed'].values[0]
            text_embed = eval(text_embed)
        except IndexError:
            if verbose:
                print('text_embedding missing Encoutered')
            text_embed = embedding_dict_mimic.iloc[-1]['embed']
            text_embed = eval(text_embed)

        condition = {'gender': gender, 'age': age, 'heart rate': hr}

        features_file_content.update({"batch": batch}) # batch_size
        features_file_content.update({"subject_id": subject_id.item()}) # 选取的样本在mimic数据集中的subject_id，注意这里不是按照读取顺序获取的id，而是DataLoader获取的数据集标注id，从Label里面拿的
        features_file_content.update({"Diagnosis": text}) # 临床文本报告原文，无prompts
        for key in condition:
            features_file_content.update({key: condition[key].item()}) # key个大小为(batch, 1, 1)的向量
        # features_file_content.update({"text_embed": str(text_embed)}) # 一个大小为(batch, 1, 1536)的向量
        # features_file_content.update({"Ori Latent": str(np.array(latent).tolist())}) # 一个大小为(1, 4, 1024 // 8)的向量

        text_embed = np.array(text_embed)
        text_embed = np.repeat(text_embed[np.newaxis, :], 1, axis=0)
        text_embed = np.repeat(text_embed[np.newaxis, :, :], batch, axis=0)

        text_embed = torch.Tensor(text_embed)
        text_embed = text_embed.to(device)
        if verbose:
            print(text_embed.shape)

        for key in condition:
            condition[key] = np.array([condition[key]])
            condition[key] = np.repeat(condition[key][np.newaxis, :], batch, axis=0)
            if verbose:
                print(condition[key].shape)
            condition[key] = torch.Tensor(condition[key])
            condition[key] = condition[key].to(device)
        if verbose:
            print(condition)

        latent = generation_from_net(diffused_model, net, batch_size=batch, device=device, text_embed=text_embed, condition=condition)
        # features_file_content.update({"Gen Latent": str(np.array(latent.cpu()).tolist())}) # 一个大小为(batch, 4, 1024 // 8)的向量

        number_str = str(index).zfill(find_power_of_ten(nums))
        save_sample_path = os.path.join(save_path, number_str +  '-' + text[:-1] + '/')
        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)

        if save_img:
            input_ = decoder(x.to(device))
            input_ = input_.squeeze(0).detach().cpu().numpy()
            wfdb.plot_items(input_, figsize=(10, 10), title="Original ECG") 
            plt.savefig(save_sample_path + 'Original ECG.png')
            plt.close()

            gen_ecg = decoder(torch.Tensor(latent))
            for j in range(batch):
                output = gen_ecg[j]

                output_ = output.squeeze(0).detach().cpu().numpy()
                wfdb.plot_items(output_, figsize=(10, 10), title="Generated ECG")
                plt.savefig(save_sample_path + f'{j} Generated ECG.png')
                plt.close()

        with open(save_sample_path + 'features.json', 'w') as json_file:
            json.dump(features_file_content, json_file, indent=4)
            # print(f"Features has been successfully written to {save_sample_path}features.json")


if __name__ == "__main__":
    nums = 100 # 选用的mimic样本数
    batch = 10 #使用[每个样本对应的condition]生成的ECG个数
    save_path = 'diversity_sample'
    device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    mimic_path = './mimic_vae_lite.pt'
    mimic_test_data = DictDataset(path=mimic_path)
    mimic_test_dataloader = DataLoader(mimic_test_data, batch_size=1, shuffle=True)

    n_channels = 4
    num_train_steps = 1000
    net = ECGconditional(num_train_steps, kernel_size=7, num_levels=5, n_channels=n_channels)
    unet_path = './checkpoints/unet_11/unet_best.pth'
    net.load_state_dict(torch.load(unet_path, map_location=device))
    net = net.to(device)

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_1/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    batch_generate_ECG(nums=nums, 
                       batch=batch, 
                       save_path=save_path, 
                       test_dataloader=mimic_test_dataloader, 
                       net=net, 
                       diffused_model=diffused_model, 
                       decoder=decoder, 
                       device=device,
                       save_img=True)

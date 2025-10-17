import torch
from vae.vae_model import VAEDecoder
from matplotlib import pyplot as plt
import numpy as np 
import wfdb
from tqdm import tqdm
import pandas as pd
import json
import math
from dataset.mimic_iv_ecg_dataset import DictDataset
from unet.unet_conditional import ECGConditional
from unet.unet_nocondition import ECGNoCondition 
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
import ecg_plot 

from scipy.interpolate import interp1d
import requests

def find_power_of_ten(number):
    if number > 0:
        power = math.log(number, 10)
        return math.ceil(power)
    else:
        return "Number must be greater than 0"
    
def generation_from_net(diffused_model: DDPMScheduler, net: ECGConditional, batch_size, device, text_embed, condition, use_vae_latent):
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
    
def resample_interp(sig, fs_in, fs_out):
    """
    基于线性拟合的差值重采样算法
    计算前后点对应的比例进行插值
    :param sig:  单导联数据，一维浮点型数组
    :param fs_in: 原始采样率，整型
    :param fs_out: 目标采样率，整型
    :return: 重采样后的数据
    """
    ts = np.array(sig).astype('float')
    t = ts.shape[0] / fs_in
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=ts.shape[0], endpoint=True)
        x_new = np.linspace(0, 1, num=int(t * fs_out), endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
    return y_new

def evaluate_age(ecgs: np.ndarray):
    # ecg: (B, L, C)
    age_list = []
    for _, ecg in enumerate(ecgs):
        # extract LEAD I
        lead_I = ecg[:, 0]
        lead_I = resample_interp(lead_I, 102.4, 500)[:5000]

        data = {
            'ecgData': lead_I.tolist(), 
            'method': 'Heartage'  
        }

        datas = json.dumps(data)
        url = "http://183.162.233.24:10081/AnyECG"  # 服务器测试
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url=url, headers=headers, data=datas)
        response_data = json.loads(response.text)
        age_list.append(response_data['data'])
    
    return age_list

def batch_generate_ECG(batch, 
                       nums,
                       save_path, 
                       test_dataset, 
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

    if not save_img:
        print("Ignore image drawing and saving...")

    for idx, (x, y) in enumerate(test_dataset):
        if idx > nums:
            break

        x, y = test_dataset[idx] 

        ori_latent = x.unsqueeze(0) 
        text = y['text'].lower()
        gender = 1 if y['gender'] == 'M' else 0
        gender = torch.tensor([gender])
        hr = torch.tensor([y['hr']])
        age = torch.tensor([y['age']])
        subject_id = y['subject_id']

        features_file_content = {}

        text_embed = y['text_embed']

        features_file_content.update({"batch": batch}) # batch_size
        features_file_content.update({"subject_id": subject_id.item()}) 
        features_file_content.update({"Diagnosis": text}) # prompts

        text_embed = np.array(text_embed)
        text_embed = np.repeat(text_embed[np.newaxis, :], 1, axis=0)
        text_embed = np.repeat(text_embed[np.newaxis, :, :], batch, axis=0)

        text_embed = torch.Tensor(text_embed).squeeze(-1)
        # features_file_content.update({"text_embed": str(text_embed.tolist())}) # 一个大小为(batch, 1, 1536)的向量
        text_embed = text_embed.to(device)
        if verbose:
            print(text_embed.shape)

        if use_condition: 
            condition = {'gender': gender, 'age': age, 'heart rate': hr}
            if verbose:
                print(condition)

            for key in condition:
                features_file_content.update({key: condition[key].item()}) # key个大小为(batch, 1, 1)的向量

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

        # features_file_content.update({"Ori Latent": str(np.array(ori_latent).tolist())}) 
        latent = generation_from_net(diffused_model, net, batch_size=batch, device=device, text_embed=text_embed, condition=condition, use_vae_latent=use_vae_latent)
        # features_file_content.update({"Gen Latent": str(np.array(latent.cpu()).tolist())}) 

        number_str = str(idx).zfill(find_power_of_ten(nums))
        save_sample_path = os.path.join(save_path, number_str)
        
        # gen_ecg: (B, L, C)
        if use_vae_latent: 
            gen_ecg = decoder(latent)
        else: 
            gen_ecg = latent.transpose(-1, -2) 

        gen_ecg = gen_ecg.detach().cpu().numpy()
        age_list = evaluate_age(gen_ecg)
        features_file_content.update({"age_list": age_list})

        input_ = decoder(ori_latent.to(device))
        input_ = input_.detach().cpu().numpy()
        bias = evaluate_age(input_)
        features_file_content.update({"bias": bias})

        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)
        if save_img:
            input_ = decoder(ori_latent.to(device))
            ecg_plot.plot(input_.squeeze(0).transpose(1, 0), 102.4)            
            plt.savefig(os.path.join(save_sample_path, 'Original ECG.png'))
            plt.close()


            for j in range(batch):
                output = gen_ecg[j]

                # wfdb.plot_items(output_, figsize=(10, 10), title=f"Generated ECG of age {age.item()}")
                ecg_plot.plot(output.transpose(1, 0), 102.4)            
                plt.savefig(os.path.join(save_sample_path, f'{j} Generated ECG a{age.item()}.png'))
                plt.close()

        with open(os.path.join(save_sample_path, 'features.json'), 'w') as json_file:
            json.dump(features_file_content, json_file, indent=4)
            # print(f"Features has been successfully written to {save_sample_path}features.json")


if __name__ == "__main__":
    nums = 250
    batch = 10 
    use_vae_latent = True 
    use_condition = True 
    test_only = True

    save_path = './exp/age'
    if not test_only:
        save_img = False
        device_str = "cuda:1"
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        mimic_path = './prerequisites/mimic_vae_lite_0_new.pt'
        mimic_test_data = DictDataset(path=mimic_path)
        # mimic_test_dataloader = DataLoader(mimic_test_data, batch_size=1, shuffle=False)

        n_channels = 4 if use_vae_latent else 12 
        num_train_steps = 1000
        if use_condition: 
            net = ECGConditional(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)
        else:
            net = ECGNoCondition(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)

        unet_path = './prerequisites/unet_all.pth'
        net.load_state_dict(torch.load(unet_path, map_location=device))
        net = net.to(device)

        diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
        diffused_model.set_timesteps(1000)

        decoder = VAEDecoder()
        vae_path = './prerequisites/vae_model.pth'
        checkpoint = torch.load(vae_path, map_location=device)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder = decoder.to(device)

        batch_generate_ECG(batch=batch, 
                        nums=nums,
                        save_path=save_path, 
                        test_dataset=mimic_test_data, 
                        net=net, 
                        diffused_model=diffused_model, 
                        decoder=decoder, 
                        device=device,
                        use_condition=use_condition, 
                        use_vae_latent=use_vae_latent, 
                        save_img=save_img)

    scatters = []
    for batch_file in os.listdir(path=save_path):
        if '.npy' in batch_file:
            continue
        features_path = os.path.join(save_path, batch_file)
        with open(features_path + '/features.json', 'r') as file:
            features_dict = json.load(file)
        
        age_list = features_dict['age_list']
        ref_age = features_dict['age']
        bias = features_dict['bias'][0]

        # mean_gen_age = sum(age_list) / len(age_list)

        age_list.sort()
        mean_gen_age = age_list[len(age_list) // 2]
        scatters.append((ref_age, mean_gen_age, bias))

    np.save(os.path.join(save_path, 'scatters_age.npy'), scatters)
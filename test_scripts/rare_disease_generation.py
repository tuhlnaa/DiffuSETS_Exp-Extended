import torch 
import numpy as np 
import pandas as pd
import wfdb 
import os 
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler 
from unet.unet_conditional import ECGconditional
from vae.vae_model import VAE_Decoder
from tqdm import tqdm 
import ecg_plot


def generation_from_net(diffused_model: DDPMScheduler, net: ECGconditional, batch_size, device, text_embed, condition, dim=128):
    net.eval()
    xi = torch.randn(batch_size, 4, dim)
    xi = xi.to(device)
    timesteps = tqdm(diffused_model.timesteps)
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

if __name__ == '__main__': 
    gen_batch = 100
    gender = 'M'
    age = 24
    hr = 90
    disease = 'brugada'
    save_path = f'./exp/{disease}/'
    device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    n_channels = 4
    num_train_steps = 1000
    net = ECGconditional(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)
    unet_path = './prerequistes/unet_all.pth'
    net.load_state_dict(torch.load(unet_path, map_location=device))
    net = net.to(device)

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = './prerequistes/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    net.eval()
    decoder.eval()

    try:
        os.makedirs(save_path) 
    except:
        pass 

    with open(f'./non_mimic_embeddings/embedding_{disease}.txt') as f:
        text_embedding = eval(f.readline())
    text_embedding = torch.tensor(text_embedding)
    text_embedding = text_embedding.repeat(gen_batch, 1, 1).to(device)
    gender_vec = 0 if gender == 'F' else 1

    condition = {'gender': [gender_vec] * gen_batch, 
                'age': [age] * gen_batch,
                'hr': [hr] * gen_batch}

    for key in condition:
        condition[key] = torch.tensor(condition[key]).reshape(gen_batch, 1, 1)
        condition[key] = condition[key].to(device)

    latents = generation_from_net(diffused_model=diffused_model, 
                                net=net, 
                                text_embed=text_embedding, 
                                condition=condition, 
                                batch_size=gen_batch, 
                                device=device) 
    
    torch.save(latents.cpu(), os.path.join(save_path, 'latent.pt')) 

    if latents.shape[0] <= 20:
        ecgs = decoder(latents)
    else:
        ecgs = [decoder(minibatch_latent) for minibatch_latent in torch.split(latents, 20)]
        ecgs = torch.concat(ecgs, dim=0)

    for idx, ecg in enumerate(ecgs):
        output_ = ecg.detach().cpu().numpy()
        # wfdb.plot_items(output_, figsize=(10, 10), title="Generated ECG.png")
        ecg_plot.plot(output_.transpose(1, 0), 102.4, title=None, columns=1, row_height=4)            
        plt.savefig(os.path.join(save_path, f'{idx} Generated ECG.png'))
        plt.close()
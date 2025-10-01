import os 
import sys
import torch 
import ecg_plot
import numpy as np 
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler 
from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from unet.unet_conditional import ECGconditional
from vae.vae_model import VAE_Decoder
from utils.inference import generation_from_net
from utils.config import init_seeds


if __name__ == '__main__': 
    batch_size = 100
    gender = 'M'
    age = 24
    hr = 90
    disease = 'brugada'
    save_path = f'./exp/{disease}/'
    device_str = "cuda:0"
    unet_path = './checkpoints/unet_all.pth'
    vae_path = './checkpoints/vae_model.pth'
    init_seeds()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    n_channels = 4
    num_train_steps = 1000
    unet = ECGconditional(num_train_steps, kernel_size=7, num_levels=7, n_channels=n_channels)
    
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet = unet.to(device)

    diffused_model = DDPMScheduler(num_train_timesteps=num_train_steps, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    unet.eval()
    decoder.eval()

    try:
        os.makedirs(save_path) 
    except:
        pass 

    text_embedding = np.load(f'./non_mimic_embeddings/embedding_{disease}.npy')
    text_embedding = torch.tensor(text_embedding)

    text_embedding = text_embedding.repeat(batch_size, 1, 1).to(device)
    gender_vec = 0 if gender == 'F' else 1

    condition_dict = {'gender': [gender_vec] * batch_size, 
                'age': [age] * batch_size,
                'hr': [hr] * batch_size}

    for key in condition_dict:
        condition_dict[key] = torch.tensor(condition_dict[key]).reshape(batch_size, 1, 1)
        condition_dict[key] = condition_dict[key].to(device)

    # Generate latent representations
    latent = generation_from_net(
        diffused_model, 
        unet, 
        batch_size, 
        device, 
        text_embed=text_embedding, 
        condition=condition_dict,
        num_channels=4,
        dim=128
    )

    torch.save(latent.cpu(), os.path.join(save_path, 'latent.pt')) 

    if latent.shape[0] <= 20:
        ecgs = decoder(latent)
    else:
        ecgs = [decoder(minibatch_latent) for minibatch_latent in torch.split(latent, 20)]
        ecgs = torch.concat(ecgs, dim=0)

    print(ecgs.shape) # [100, 1024, 12]

    for idx, ecg in enumerate(ecgs):
        output_ = ecg.detach().cpu().numpy()
        # wfdb.plot_items(output_, figsize=(10, 10), title="Generated ECG.png")
        ecg_plot.plot(output_.transpose(1, 0), 102.4, title=None, columns=1, row_height=4)            
        plt.savefig(os.path.join(save_path, f'{idx} Generated ECG.png'), dpi=200)
        plt.close()
        break
import torch
import argparse 
import json
from vae.vae_model import VAE_Decoder
from unet.unet_conditional import ECGconditional
from unet.unet_nocondition import ECGnocondition
from diffusers import DDPMScheduler
from utils.inference import batch_generate_ECG 

def parse_arg():
    parser = argparse.ArgumentParser(description='DiffuSETS Inference') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

def main():

    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = json.load(f) 

    settings = config['inference_setting']
    h_ = config['hyper_para']

    condition = config['train_meta']['condition']
    if condition:
        unet = ECGconditional(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'], num_levels=h_['unet_num_level'], n_channels=4)
    else: 
        unet = ECGnocondition(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'], num_levels=h_['unet_num_level'], n_channels=4)
    unet_path = settings['unet_path']
    unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

    diffused_model = DDPMScheduler(num_train_timesteps=h_['num_train_steps'], beta_start=h_['beta_start'], beta_end=h_['beta_end'])
    diffused_model.set_timesteps(settings['inference_timestep'])

    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_1/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])

    batch_generate_ECG(settings=settings, 
                       unet=unet, 
                       diffused_model=diffused_model, 
                       decoder=decoder, 
                       condition=condition)

if __name__ == "__main__":
    main() 
import torch
from torch.utils.data import DataLoader
from vae.vae_model import VAEDecoder, VAE_Encoder, loss_function

import os
from dataset.mimic_iv_ecg_dataset import MIMIC_IV_ECG_Dataset

import logging

if __name__ == '__main__':

    k_max = 0
    for item in os.listdir('./checkpoints'):
        if "vae_" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/vae_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'vae_{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    H_ = {
        'lr': 1e-4, 
        'batch_size': 256, 
        'epochs': 10
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device('cuda:6')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    
    path = '/data1_science/1shared/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    mimic_train = MIMIC_IV_ECG_Dataset(path, usage='all', resample_length=1024)
    train_dataloader = DataLoader(mimic_train, batch_size=H_['batch_size'])

    encoder = VAE_Encoder().to(device)
    decoder = VAEDecoder().to(device)

    def train_loop(dataloader, encoder, decoder, loss_fn, optimizer, scheduler, kld_weight):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        encoder.train()
        decoder.train()

        for batch, (X, _) in enumerate(dataloader):
            torch.autograd.set_detect_anomaly(True)
            # Compute prediction and loss
            # X: (B, L, C)
            X = X.to(device)
            # latent_shape (B, 4, L/8)
            z, mean, log_var = encoder(X)

            recons = decoder(z)
            loss = loss_fn(recons, X, mean, log_var, kld_weight)

            # Backpropagation
            loss['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            if batch % 50 == 0:
                loss, current = loss, (batch + 1) * len(X)
                logger.info(f"loss: {loss['loss']:>7f}  mse: {loss['mse']:>7f}  KLd: {loss['KLD']:>7f}  [{current:>5d}/{size:>5d}]")

    loss_fn = loss_function
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=H_['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, 
                    epochs=H_['epochs'], steps_per_epoch=len(train_dataloader))

    for t in range(H_['epochs']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        kld_weights = torch.arange(1, H_['epochs'] + 1) / H_['epochs']
        train_loop(train_dataloader, encoder, decoder, loss_fn, optimizer, scheduler, kld_weights[t % 10])
        if is_save and t > 5:
            model_states = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
            save_path = os.path.join(save_weights_path, f"VAE_model_ep{t}.pth")
            torch.save(model_states, save_path)
    logger.info("done!")

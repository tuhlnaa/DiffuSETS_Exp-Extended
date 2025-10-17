import torch
from vae.vae_model import VAE_Encoder, VAEDecoder, loss_function
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.mimic_iv_ecg_dataset import MIMIC_IV_ECG_Dataset

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    vae_path = './checkpoints/vae_1/VAE_model_ep9.pth'
    vae_weight_dict = torch.load(vae_path, map_location=device) 
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_weight_dict['encoder'])
    encoder.to(device)
    decoder = VAEDecoder()
    decoder.load_state_dict(vae_weight_dict['decoder'])
    decoder.to(device)

    path = '/data1_science/1shared/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    mimic_test = MIMIC_IV_ECG_Dataset(path, usage='all', resample_length=1024)
    test_dataloader = DataLoader(mimic_test, batch_size=512)

    @torch.no_grad()
    def test_loop(dataloader, encoder, decoder, loss_fn):
        encoder.eval()
        decoder.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, _ in tqdm(dataloader):
                X = X.to(device)
                z, mean, log_var = encoder(X)
                recons = decoder(z)
                test_loss += loss_fn(recons, X, mean, log_var)['loss']

        test_loss /= num_batches
        print(f"Test Error Avg loss: {test_loss:>8f} \n")

    test_loop(test_dataloader, encoder, decoder, loss_function)

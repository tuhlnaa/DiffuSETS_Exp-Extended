import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from clip.clip_model import CLIP
from dataset.ptbxl_dataset import PtbxlDataset, PtbxlDataset_VAE
from dataset.mimic_iv_ecg_dataset import VAE_MIMIC_IV_ECG_Dataset
from vae.vae_model import VAE_Decoder

import os
import logging

embedding_dict_ptbxl = pd.read_csv('/data/0shared/chenjiabo/DiffuSETS/data/ptbxl_database_embed.csv', low_memory=False)[['ecg_id', 'text_embed']]
original_sheet = pd.read_csv('/data/0shared/laiyongfan/data_text2ecg/ptb-xl/ptbxl_database.csv', low_memory=False)
embedding_dict_ptbxl = pd.merge(embedding_dict_ptbxl, original_sheet)

def fetch_text_embedding_ptbxl(text:str):
    text = text.split('|')[0]
    text = text.replace('The report of the ECG is that ', '')
    try:
        text_embed = embedding_dict_ptbxl.loc[embedding_dict_ptbxl['report'] == text, 'text_embed'].values[0]
        text_embed = eval(text_embed)
    except IndexError:
        text_embed = [0] * 1536
    # print(text_embed)
    return torch.tensor(text_embed)

embedding_dict_mimic = pd.read_csv('/data/0shared/chenjiabo/DiffuSETS/data/mimic_iv_text_embed.csv')

def fetch_text_embedding_mimic_report_0(text: str):
    text = text.split('|')[0]
    if len(text) > 0 and text[-1] != '.':
        text += '.'
    try:
        text_embed = embedding_dict_mimic.loc[embedding_dict_mimic['text'] == text, 'embed'].values[0]
        text_embed = eval(text_embed)
    except IndexError:
        text_embed = embedding_dict_mimic.iloc[-1]['embed']
        text_embed = eval(text_embed)
        # print(1, text)
    return torch.tensor(text_embed)
    

def train_batch(ecgs, text_embeddings, model, device, criterion, optimizer):
    ecgs, text_embeddings = ecgs.to(device), text_embeddings.to(device)
    
    # Forward pass 
    logits_per_ecg, logits_per_text = model(ecgs, text_embeddings)
    
    # Create labels
    batch_size = ecgs.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_ecg = criterion(logits_per_ecg, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_ecg + loss_txt)/2 # avg. ecg and txt loss

    # Backward pass 
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_loop(dataloader, fetch_func, model, loss_fn, optimizer, device, decoder=None):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        texts = y['text']
        texts = [fetch_func(x) for x in texts]
        text_embedding = torch.stack(texts)

        if decoder:
            X = X.to(device)
            X = decoder(X)

        loss = train_batch(ecgs=X, text_embeddings=text_embedding, model=model, device=device, criterion=loss_fn, optimizer=optimizer)
        total_loss += loss

        if batch % 100 == 0:
            loss, current = loss, (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def eval_score(dataloader, fetch_func, model, device, decoder=None):
    model.eval()
    decoder.eval()

    total_clip_score = 0
    for batch, (X, y) in enumerate(dataloader):
        texts = y['text']
        texts = [fetch_func(x) for x in texts]
        text_embedding = torch.stack(texts).to(device)

        X = X.to(device)
        if decoder:
            X = decoder(X)

        signal_embedding = clip_model.encode_signal(X)

        # signal features: (B, embed_dim)
        signal_features = clip_model.ecg_projector(signal_embedding)
        # text features:  (B, embed_dim)
        text_features = clip_model.text_projector(text_embedding)

        # normalized features
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        batch_clip_score = torch.trace(signal_features @ text_features.t()) 

        total_clip_score += batch_clip_score

    return total_clip_score / len(dataloader.dataset)




if __name__ == '__main__':

    k_max = 0
    for item in os.listdir('./checkpoints'):
        if "clip" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/clip_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'clip{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    H_ = {
        'embed_dim': 64, 
        'lr': 1e-5, 
        'batch_size': 256, 
        'epochs': 50, 
        'load_from_pretrain': False
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    ptb_path = '/data/0shared/laiyongfan/data_text2ecg/ptb-xl/'
    ptb_vae_path = '/data/0shared/laiyongfan/data_text2ecg/ptb-xl_vae'
    mimiv_vae_path = '/data/0shared/laiyongfan/data_text2ecg/mimic_vae'
    # train_dataset = PtbxlDataset(ptbxl_path=ptb_path, sampling_rate=500, use_all=True, combine_diagnostic=False) 
    train_dataset = VAE_MIMIC_IV_ECG_Dataset(mimiv_vae_path)
    # train_dataset = PtbxlDataset_VAE(path=ptb_vae_path)
    # test_dataset = PtbxlDataset(ptbxl_path=ptb_path, sampling_rate=500, is_train=False, combine_diagnostic=False)
    test_dataset = VAE_MIMIC_IV_ECG_Dataset(mimiv_vae_path, usage='test')
    train_dataloader = DataLoader(train_dataset, batch_size=H_['batch_size'], shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    decoder = None
    decoder = VAE_Decoder()
    vae_path = '/data/0shared/laiyongfan/data_text2ecg/models/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    clip_model = CLIP(embed_dim=H_['embed_dim'])

    if H_['load_from_pretrain']:
        pretrain_model_root = './checkpoints/clip_3/CLIP_model_ep12.pth'
        pretrain_model_weight = torch.load(pretrain_model_root, map_location=device)
        logger.info(f"Loading from {pretrain_model_root}")
        clip_model.load_state_dict(pretrain_model_weight)

    clip_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=H_['lr'])

    for t in range(H_['epochs']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        kld_weights = torch.arange(1, H_['epochs'] + 1) / H_['epochs']
        # Setting decoder to None if using original ECG
        epoch_avg_loss = train_loop(train_dataloader, fetch_text_embedding_mimic_report_0, clip_model, loss_fn, optimizer, device, decoder=decoder)
        logger.info(f"Evaluating training clip score...")
        # Setting decoder to None if using original ECG
        epoch_avg_clip_score = eval_score(test_dataloader, fetch_text_embedding_mimic_report_0, clip_model, device, decoder=decoder)
        logger.info(f"Epoch {t+1} Loss: {epoch_avg_loss} EVAL CLIP Score: {epoch_avg_clip_score}")
        if is_save and t > 10:
            save_path = os.path.join(save_weights_path, f"CLIP_model_ep{t+1}.pth")
            torch.save(clip_model.state_dict(), save_path)
    logger.info("done!")

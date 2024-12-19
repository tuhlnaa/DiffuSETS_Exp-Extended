import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

from dataset.mimic_iv_ecg_dataset import DictDataset
from classification.classifier import ResNetECG, TransformerECG 
from vae.vae_model import VAE_Decoder 

from sklearn.metrics import confusion_matrix, f1_score 

import os
import logging

def train_batch(ecgs, labels, model, criterion, optimizer):
    # Forward pass 
    logit = model(ecgs) 
    
    # Compute loss
    loss = criterion(logit, labels)

    # Backward pass 
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_loop(dataloader, model, loss_fn, optimizer, device, decoder=None):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        if decoder:
            X = X.to(device)
            X = decoder(X) 
        else: 
            X = X.to(device) 
        labels = y['label']
        labels = labels.to(device)

        loss = train_batch(ecgs=X, labels=labels, model=model,  criterion=loss_fn, optimizer=optimizer)
        total_loss += loss.detach()

        if batch % 25 == 0:
            loss, current = loss, (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / size

@torch.no_grad()
def vali_loop(dataloader, model, loss_fn, device, decoder=None):
    size = len(dataloader.dataset)
    model.eval()
    decoder.eval()

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        if decoder:
            X = decoder(X)
        labels = y['label'].to(device) 

        logit = model(X) 
        loss = loss_fn(logit, labels) 
    
    total_loss += loss 
    return total_loss / size

@torch.no_grad()
def test_loop(dataloader, model, device, decoder): 
    size = len(dataloader.dataset) 
    model.eval() 
    decoder.eval() 

    all_label = [] 
    all_pred = [] 
    for batch, (X, y) in enumerate(dataloader): 
        X = X.to(device) 
        if decoder: 
            X = decoder(X) 
        labels = y['label'].numpy() 
        all_label.extend(labels) 

        logit = model(X) 
        pred = torch.argmax(logit, dim=-1).cpu().numpy() 

        all_pred.extend(pred)

    acc = np.sum(np.equal(all_pred, all_label)) / size 
    cm = confusion_matrix(all_label, all_pred) 
    f1 = f1_score(all_label, all_pred, average='macro')

    return acc, f1, cm 


if __name__ == '__main__':

    k_max = 0
    exp_type = 'normal'
    model_type = 'Transformer' 
    is_cmp = ''
    no_weight = False
    assert exp_type in ['normal', 'af', 'pvc']
    assert model_type in ['ResNet', 'Transformer']
    assert (is_cmp == '') or (is_cmp == '_cmp')
    for item in os.listdir('./checkpoints'):
        if f"clf_{exp_type}_{model_type}" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = f"./checkpoints/clf_{exp_type}_{model_type}_{k_max + 1}"

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f'clf_{exp_type}_{model_type}_{k_max + 1}')
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    H_ = {
        'lr': 1e-3,  
        'batch_size': 256, 
        'epochs': 10, 
        'load_from_pretrain': False, 
        'exp_type': exp_type, 
        'model_type': model_type, 
        'is_cmp': is_cmp, 
        'no_weight': no_weight
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    train_dataset_path = f'./prerequisites/mimic_vae_clf_{exp_type}_train{is_cmp}.pt'
    vali_dataset_path = f'./prerequisites/mimic_vae_clf_{exp_type}_valid.pt'
    test_dataset_path = f'./prerequisites/mimic_vae_clf_{exp_type}_test.pt'
    train_dataset = DictDataset(train_dataset_path)
    vali_dataset = DictDataset(vali_dataset_path)
    test_dataset = DictDataset(test_dataset_path) 
    train_dataloader = DataLoader(train_dataset, batch_size=H_['batch_size'], shuffle=True) 
    vali_dataloader = DataLoader(vali_dataset, batch_size=H_['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=H_['batch_size'], shuffle=True)

    decoder = None
    decoder = VAE_Decoder()
    vae_path = './checkpoints/vae_1/vae_model.pth'
    checkpoint = torch.load(vae_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)

    if model_type == 'ResNet': 
        clf_model = ResNetECG(num_classes=2, ecg_channels=12)
    else:
        clf_model = TransformerECG()

    clf_model.to(device)
    if is_cmp or no_weight: 
        class_weight = torch.tensor([1, 1]) 
    elif exp_type == 'af': 
        class_weight = torch.tensor([10, 1])
    elif exp_type == 'pvc': 
        class_weight = torch.tensor([15, 1]) 
    else:
        class_weight = torch.tensor([2, 1])
    class_weight = class_weight.to(device=device, dtype=torch.float) 

    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=H_['lr'], weight_decay=3e-4)

    min_loss = float("inf")
    for t in range(H_['epochs']):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # Setting decoder to None if using original ECG
        epoch_train_loss = train_loop(train_dataloader, clf_model, loss_fn, optimizer, device, decoder=decoder)
        logger.info(f"Evaluating validation loss...")
        # Setting decoder to None if using original ECG
        epoch_vali_loss = vali_loop(test_dataloader, clf_model, loss_fn, device, decoder=decoder)
        logger.info(f"Epoch {t+1} Train Loss: {epoch_train_loss} Vali Loss: {epoch_vali_loss}")
        if epoch_vali_loss < min_loss and t + 1 >= H_['epochs'] * 0.5:
            save_path = os.path.join(save_weights_path, "clf_best.pth")
            torch.save(clf_model.state_dict(), save_path)
            logger.info("clf_best has been saved")
            min_loss = epoch_vali_loss
        # if (t + 1) % 10 == 0:
        #     torch.save(clf_model.state_dict(), os.path.join(save_weights_path, f'clf_model_ep{t + 1}.pth'))
    
    # load best model 
    logger.info(f"Loading the best {model_type} model")
    model_best = clf_model.load_state_dict(torch.load(save_path, map_location=device))
    acc, f1, cm = test_loop(test_dataloader, clf_model, device, decoder) 
    np.save(os.path.join(save_weights_path, 'cm.npy'), cm) 
    logger.info(f"{cm[0][0]}\t{cm[0][1]}\n{cm[1][0]}\t{cm[1][1]}")
    logger.info(f"Acc: {acc}\n F1: {f1}")
    logger.info("done!")

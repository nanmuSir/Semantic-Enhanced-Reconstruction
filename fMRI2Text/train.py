import torch
import numpy as np

def train(model, dataloader, criterion, optimizer):
    model.train()
    for batch in dataloader:
        fMRI_data, text_data = batch
        optimizer.zero_grad()
        output = model(fMRI_data)
        loss = criterion(output, text_data)
        loss.backward()
        optimizer.step()

def infer(model, fMRI_data):
    model.eval()
    with torch.no_grad():
        latent = model.encoder(fMRI_data)
        seq_output = model.seq_model(latent.unsqueeze(1))
    return seq_output

def save_latent_features(encoder, dataloader, save_path):
    encoder.eval()
    features = []
    with torch.no_grad():
        for (fMRI_data, _) in dataloader:
            latent = encoder(fMRI_data)
            features.append(latent.cpu().numpy())
    np.save(save_path, np.concatenate(features, axis=0))
    print('fmri feature saved')

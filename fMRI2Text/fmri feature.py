from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset import FMRI_Dataset

# Define the Encoder model (Assuming nn.Module is imported)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv3d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv3d(64, 128, 3, 2, 1)

        # Use a dummy forward pass to calculate the flattened feature size
        self._compute_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 512)

    def _compute_flattened_size(self):
        dummy_input = torch.zeros(1, 1, 80, 104, 80)
        output = self.conv3(self.conv2(self.conv1(dummy_input)))
        self.flattened_size = output.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Dataset class, FMRI_Dataset, already defined

# Instantiate the dataset
subjects_directories = [
    '/home/data/ZH/NSD/nsddata_betas/ppdata/subj01/2%smooth',
    '/home/data/ZH/NSD/nsddata_betas/ppdata/subj02/2%smooth',
    '/home/data/ZH/NSD/nsddata_betas/ppdata/subj05/2%smooth',
    '/home/data/ZH/NSD/nsddata_betas/ppdata/subj07/2%smooth',
]
dataset = FMRI_Dataset(subjects_directories)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder().to(device)

# Extract and save features
feature_list = []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for batch in tqdm(dataloader, desc='Extracting Features', unit='batch'):
        batch = batch.to(device)
        features = model(batch)
        feature_list.append(features.cpu().numpy())

# Convert the feature list to a numpy array and save it
features_array = np.concatenate(feature_list, axis=0)
np.save("fmri_features.npy", features_array)

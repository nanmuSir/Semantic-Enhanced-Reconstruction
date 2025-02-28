import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset

class FMRI_Dataset(Dataset):
    def __init__(self, directories):
        self.files = []
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.nii') or file.endswith('.nii.gz'):
                        self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        nii_data = nib.load(file_path).get_fdata()
        current_shape = nii_data.shape

        desired_shape = (80, 104, 80)

        # 初始化裁剪和填充
        pad = [(0, 0)] * 3
        crop = [slice(None)] * 3

        for i in range(3):
            if current_shape[i] < desired_shape[i]:
                total_pad = desired_shape[i] - current_shape[i]
                pad[i] = (total_pad // 2, total_pad - total_pad // 2)
            elif current_shape[i] > desired_shape[i]:
                start = (current_shape[i] - desired_shape[i]) // 2
                end = start + desired_shape[i]
                crop[i] = slice(start, end)

        cropped_data = nii_data[crop[0], crop[1], crop[2]]
        data_tensor = torch.from_numpy(cropped_data).float()
        pad_tuple = (pad[2][0], pad[2][1], pad[1][0], pad[1][1], pad[0][0], pad[0][1])
        padded_data = F.pad(data_tensor, pad_tuple, 'constant', 0)
        min_val = torch.min(padded_data)
        max_val = torch.max(padded_data)
        normalized_data_tensor = (padded_data - min_val) / (max_val - min_val)
        input_tensor = normalized_data_tensor.unsqueeze(0)

        return input_tensor

# Define the directories for subjects
subjects_directories = [
    '/home/data/ZH/NSD/nsddata_betas/ppdata/subj01/func1pt8mm'
    # '/home/data/ZH/NSD/nsddata_betas/ppdata/subj02/func1pt8mm',
    # '/home/data/ZH/NSD/nsddata_betas/ppdata/subj05/func1pt8mm',
    # '/home/data/ZH/NSD/nsddata_betas/ppdata/subj07/func1pt8mm'
]

# Instantiate dataset
dataset = FMRI_Dataset(subjects_directories)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


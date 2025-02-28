import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class ImagefMRIDataset(Dataset):
    def __init__(self, img_dir, fmri_dir, transform=None):
        self.img_dir = img_dir
        self.fmri_dir = fmri_dir
        self.img_files = os.listdir(img_dir)
        self.fmri_files = os.listdir(fmri_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        fmri_path = os.path.join(self.fmri_dir, self.fmri_files[idx])

        image = Image.open(img_path).convert("RGB")
        fmri = np.load(fmri_path)

        if self.transform:
            image = self.transform(image)

        fmri = torch.tensor(fmri, dtype=torch.float32)

        return image, fmri

class ViTModule(nn.Module):
    def __init__(self, img_size=425, patch_size=16, embed_dim=512):
        super(ViTModule, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.vit = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Linear(self.num_patches * self.embed_dim, self.embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x -> [batch_size, 3, 425, 425]
        vit_output = self.vit(x)
        # vit_output -> [batch_size, 512, num_patches]
        return vit_output

class FourierEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(FourierEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # 期望输入为 [batch_size, 512]，使用正弦和余弦编码
        pos_enc = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return pos_enc

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RidgeRegression, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, brain_map, img_features):
        combined = torch.cat([brain_map, img_features], dim=1)
        return self.fc(combined)

class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

class ImageReconstructionModel(nn.Module):
    def __init__(self):
        super(ImageReconstructionModel, self).__init__()
        self.vit = ViTModule()
        self.fourier_embedding = FourierEmbedding(embed_dim=512)
        self.mlp = MLP(input_dim=512, output_dim=512)
        self.ridge_regression = RidgeRegression(input_dim=1024, output_dim=512)
        self.cnn_decoder = CNNDecoder()

    def forward(self, image, brain_map):
        # 1. 提取图像特征
        img_features = self.vit(image)  # 输出为 [batch_size, 512, num_patches]
        img_features = img_features.mean(dim=2)  # 聚合为 [batch_size, 512]

        # 2. 通过傅里叶编码和MLP
        img_features = self.fourier_embedding(img_features)
        img_features = self.mlp(img_features)  # 输出为 [batch_size, 512]

        # 3. 将图像特征和fMRI特征结合
        combined_features = self.ridge_regression(brain_map, img_features)  # [batch_size, 512]

        # 4. reshape 到CNN decoder输入形状
        img_features_reshaped = combined_features.view(-1, 512, 26, 26)  # [batch_size, 512, 26, 26]

        # 5. 通过CNN decoder生成图像
        reconstructed_image = self.cnn_decoder(img_features_reshaped)
        return reconstructed_image

def train(model, dataloader, optimizer, criterion, epochs=10, save_path="model.pth"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for brain_map, stimulus_img in dataloader:
            optimizer.zero_grad()
            output_img = model(stimulus_img, brain_map)
            loss = criterion(output_img, stimulus_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader)}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def prepare_dataloader(img_dir, fmri_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((425, 425)),
        transforms.ToTensor(),
    ])

    dataset = ImageFMriDataset(img_dir=img_dir, fmri_dir=fmri_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    img_dir = "/home/work/ZH/brain-diffuser-main/data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    fmri_dir = ""

    dataloader = prepare_dataloader(img_dir, fmri_dir)

    model = ImageReconstructionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 开始训练
    train(model, dataloader, optimizer, criterion, epochs=10, save_path="image_reconstruction_model.pth")

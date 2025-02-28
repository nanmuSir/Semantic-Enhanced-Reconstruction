import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as tvmodels
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import clip

images_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的输入图像文件夹路径
feats_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的输出特征文件夹路径

# 确保输出文件夹存在
if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)

class BatchGeneratorExternalImages(Dataset):
    def __init__(self, data_path='', prefix='', net_name='clip'):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name

        if self.net_name == 'clip':
            self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                  std=[0.26862954, 0.26130258, 0.27577711])
        else:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]
        print(f"Found {len(self.image_files)} images in {self.data_path}")  # 调试信息

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        img = Image.open(img_path)
        #print(f"Loaded image: {img_path}")  # 调试信息
        img = T.functional.resize(img, (224, 224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return len(self.image_files)

global feat_list
feat_list = []

def fn(module, inputs, outputs):
    print(f"Hook called for module: {module}")  # 调试信息
    print(f"Inputs shape: {inputs[0].shape}, Outputs shape: {outputs.shape}")  # 调试信息
    feat_list.append(outputs.cpu().numpy())

# 设置模型和层的列表
net_list = [
    ('inceptionv3', 'avgpool'),
    ('clip', 'final'),
    ('alexnet', 2),
    ('alexnet', 5),
    ('efficientnet', 'avgpool'),
    ('swav', 'avgpool')
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

for net_name, layer in net_list:
    feat_list = []
    print(f"Processing {net_name}, layer: {layer}")
    dataset = BatchGeneratorExternalImages(data_path=images_dir, net_name=net_name, prefix='')
    loader = DataLoader(dataset, batch_size, shuffle=False)

    if net_name == 'inceptionv3':
        net = tvmodels.inception_v3(pretrained=True)
        if layer == 'avgpool':
            net.avgpool.register_forward_hook(fn)
            print("Registered hook for avgpool")  # 调试信息
        elif layer == 'lastconv':
            net.Mixed_7c.register_forward_hook(fn)
            print("Registered hook for lastconv")  # 调试信息

    elif net_name == 'alexnet':
        net = tvmodels.alexnet(pretrained=True)
        if layer == 2:
            net.features[4].register_forward_hook(fn)
            print("Registered hook for layer 2")  # 调试信息
        elif layer == 5:
            net.features[11].register_forward_hook(fn)
            print("Registered hook for layer 5")  # 调试信息

    elif net_name == 'clip':
        model, _ = clip.load("ViT-L/14", device=device, jit=False)
        net = model.visual
        net = net.to(torch.float32)  # 将模型转换为float32
        if layer == 'final':
            net.register_forward_hook(fn)
            print("Registered hook for final")  # 调试信息

    elif net_name == 'efficientnet':
        net = tvmodels.efficientnet_b1(pretrained=True)
        net.avgpool.register_forward_hook(fn)
        print("Registered hook for avgpool")  # 调试信息

    elif net_name == 'swav':
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50', pretrained=True)
        net.avgpool.register_forward_hook(fn)
        print("Registered hook for avgpool")  # 调试信息

    net.eval()
    net.to(device)

    with torch.no_grad():
        for i, x in enumerate(loader):
            print(f"Processing batch {i * batch_size}, batch size: {x.shape}")  # 调试信息
            x = x.to(device).to(torch.float32)  # 确保输入数据类型为float32
            outputs = net(x)  # 明确调用模型进行前向传播
            print(f"Model output shape: {outputs.shape}")  # 调试信息

    if feat_list:  # 检查feat_list是否为空
        if net_name == 'clip':
            if layer in [7, 12]:
                feat_list = np.concatenate(feat_list, axis=1).transpose((1, 0, 2))
            else:
                feat_list = np.concatenate(feat_list)
        else:
            feat_list = np.concatenate(feat_list)

        file_name = f'{feats_dir}/{net_name}_{layer}.npy'
        np.save(file_name, feat_list)
    else:
        print(f"No features extracted for {net_name}, layer: {layer}")

print("Feature extraction completed.")




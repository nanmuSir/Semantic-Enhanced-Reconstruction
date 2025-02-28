import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as tvmodels
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

# 设置输入文件夹路径
images_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的输入图像文件夹路径
feats_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的输出特征文件夹路径

# 确保输出文件夹存在
if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)

class BatchGeneratorExternalImages(Dataset):
    def __init__(self, data_path=''):
        self.data_path = data_path

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]
        print(f"Found {len(self.image_files)} images in {self.data_path}")  # 调试信息

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = T.functional.resize(img, (224, 224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return len(self.image_files)

global feat_list
feat_list = []

def fn(module, inputs, outputs):
    #print(f"Hook called for module: {module}")  # 调试信息
    print(f"Inputs shape: {inputs[0].shape}, Outputs shape: {outputs.shape}")  # 调试信息
    feat_list.append(outputs.cpu().numpy())

# 设置ResNet50模型和层的列表
resnet_model = tvmodels.resnet50(pretrained=True)

# 可以通过以下列表设置要提取特征的层
layer_dict = {
    'layer1': resnet_model.layer1,
    'layer2': resnet_model.layer2,
    'layer3': resnet_model.layer3,
    'layer4': resnet_model.layer4,
    'avgpool': resnet_model.avgpool,
    'fc': resnet_model.fc,
}

selected_layer = 'fc'  # 修改为您想要提取特征的层

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

# 清空特征列表
feat_list = []

print(f"Processing ResNet50, layer: {selected_layer}")
dataset = BatchGeneratorExternalImages(data_path=images_dir)
loader = DataLoader(dataset, batch_size, shuffle=False)

# 注册hook
layer_dict[selected_layer].register_forward_hook(fn)
print(f"Registered hook for {selected_layer}")  # 调试信息

resnet_model.eval()
resnet_model.to(device)

with torch.no_grad():
    for i, x in enumerate(loader):
        print(f"Processing batch {i * batch_size}, batch size: {x.shape}")  # 调试信息
        x = x.to(device).to(torch.float32)  # 确保输入数据类型为float32
        outputs = resnet_model(x)  # 明确调用模型进行前向传播
        print(f"Model output shape: {outputs.shape}")  # 调试信息

if feat_list:  # 检查feat_list是否为空
    feat_list = np.concatenate(feat_list)
    file_name = f'{feats_dir}/resnet50_{selected_layer}.npy'
    np.save(file_name, feat_list)
else:
    print(f"No features extracted for ResNet50, layer: {selected_layer}")

print("Feature extraction completed.")

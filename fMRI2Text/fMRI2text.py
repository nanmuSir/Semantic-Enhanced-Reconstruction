import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 定义3D自动编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 11 * 13, 512)  # Example dimensions for the bottleneck layer
                            # 10 * 13 * 10
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        return x

# TransformerModel 来学习特征和向量的关系，并生成向量
class TransformerSeqModel(nn.Module):
    def __init__(self):
        super(TransformerSeqModel, self).__init__()
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Linear(512, 200)  # Vocabulary size for text generation

    def forward(self, x):
        tgt = torch.zeros_like(x)  # Placeholder target sequence (e.g., during training)
        output = self.transformer(x, tgt)
        output = self.fc(output)
        return output

class fMRITextModel(nn.Module):
    def __init__(self):
        super(fMRITextModel, self).__init__()
        self.encoder = Encoder()
        #self.decoder = Decoder()
        self.seq_model = TransformerSeqModel()

    def forward(self, x):
        latent = self.encoder(x)
        seq_output = self.seq_model(latent.unsqueeze(1))  # Add sequence dimension
        #text_output = self.decoder(seq_output)
        return seq_output

# 定义训练和推理过程
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
        text_output = model.decoder(seq_output)
    return text_output

# 读取fMRI数据
def load_fMRI_data(folder_path):
    file_names = sorted(os.listdir(folder_path))
    data = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        fMRI_volume = np.load(file_path)  # 根据实际情况调整文件读取方法
        data.append(fMRI_volume)
    return np.array(data)

# 读取文本数据
def load_text_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [item['description'] for item in data]  # 假设JSON中有description字段

# 数据预处理
def preprocess_data(fMRI_data, text_data, doc2vec_model):
    # 标准化fMRI数据
    fMRI_data = (fMRI_data - np.mean(fMRI_data, axis=(1, 2, 3), keepdims=True)) / np.std(fMRI_data, axis=(1, 2, 3),
                                                                                         keepdims=True)
    # 将fMRI数据转换为PyTorch张量
    fMRI_data_tensor = torch.tensor(fMRI_data, dtype=torch.float32)

    # 使用Doc2Vec将文本描述向量化
    text_vectors = [doc2vec_model.infer_vector(text.split()) for text in text_data]
    text_vectors_tensor = torch.tensor(text_vectors, dtype=torch.float32)

    return fMRI_data_tensor, text_vectors_tensor

# 加载预训练Doc2Vec模型
def load_doc2vec_model(model_path):
    return Doc2Vec.load(model_path)

# 加载数据
fMRI_folder_path = 'path_to_fMRI_data_folder'
text_json_path = 'path_to_text_data.json'
doc2vec_model_path = 'path_to_pretrained_doc2vec_model.bin'

doc2vec_model = load_doc2vec_model(doc2vec_model_path)
fMRI_data = load_fMRI_data(fMRI_folder_path)
text_data = load_text_data(text_json_path)
fMRI_data_tensor, text_vectors_tensor = preprocess_data(fMRI_data, text_data, doc2vec_model)

dataset = TensorDataset(fMRI_data_tensor, text_vectors_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = fMRITextModel()

# 选择损失函数
criterion = nn.MSELoss()  # 均方误差损失
# 其他可能的损失函数
# criterion = nn.CrossEntropyLoss()  # 适用于分类任务
# criterion = nn.L1Loss()  # 绝对误差损失

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train(model, dataloader, criterion, optimizer)

# 预测新fMRI数据
def predict(model, new_fMRI_folder_path, doc2vec_model):
    new_fMRI_data = load_fMRI_data(new_fMRI_folder_path)
    new_fMRI_data = (new_fMRI_data - np.mean(new_fMRI_data, axis=(1, 2, 3), keepdims=True)) / np.std(new_fMRI_data,
                                                                                                     axis=(1, 2, 3),
                                                                                                     keepdims=True)
    new_fMRI_data_tensor = torch.tensor(new_fMRI_data, dtype=torch.float32)

    # 模型推理
    model.eval()
    with torch.no_grad():
        latent = model.encoder(new_fMRI_data_tensor)
        seq_output = model.seq_model(latent.unsqueeze(1))
        predicted_vectors = model.decoder(seq_output)

    # 使用Doc2Vec模型将预测向量转换为自然语言描述
    # 并不是去训练集中找最相似的描述语句
    predicted_vectors = predicted_vectors.numpy()
    predicted_texts = [doc2vec_model.dv.most_similar([vec], topn=1)[0][0] for vec in predicted_vectors]

    return predicted_texts

# 预测新数据
new_fMRI_folder_path = 'path_to_new_fMRI_data_folder'
predicted_texts = predict(model, new_fMRI_folder_path, doc2vec_model)
for text in predicted_texts:
    print(text)

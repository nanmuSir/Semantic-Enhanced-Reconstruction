import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv3d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv3d(64, 128, 3, 2, 1)
        self.fc1 = nn.Linear(128 * 12 * 11 * 13, 512)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class TransformerSeqModel(nn.Module):
    def __init__(self):
        super(TransformerSeqModel, self).__init__()
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        tgt = torch.zeros_like(x)
        output = self.transformer(x, tgt)
        output = self.fc(output)
        return output

class fMRITextModel(nn.Module):
    def __init__(self):
        super(fMRITextModel, self).__init__()
        self.encoder = Encoder()
        self.seq_model = TransformerSeqModel()

    def forward(self, x):
        latent = self.encoder(x)
        seq_output = self.seq_model(latent.unsqueeze(1))
        return seq_output

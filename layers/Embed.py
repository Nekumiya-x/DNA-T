
import numpy as np
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, c_in, d_model=128, max_len=200, dropout=0.1):
        super(Embedding, self).__init__()

        self.time_emb = TimeEmbedding(d_model=d_model, max_len=max_len)

        self.conv1d_x = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.conv1d_m = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.conv1d_d = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True))

        self.conv1d_mdt = nn.Sequential(
            nn.Conv1d(in_channels=d_model * 3, out_channels=d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True))


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, m, d, t):
        # B, L, E --> B, L, D

        t = self.time_emb(t).to(x.device)
        m = self.conv1d_m(m.permute(0, 2, 1)).permute(0, 2, 1)
        d = self.conv1d_d(d.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1d_x(x.permute(0, 2, 1)).permute(0, 2, 1)

        mdt = torch.cat([m, d, t], dim=-1)
        mdt = self.conv1d_mdt(mdt.permute(0, 2, 1)).permute(0, 2, 1)

        out = torch.sigmoid(mdt) * x

        return self.dropout(out), self.dropout(mdt)



class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(TimeEmbedding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len

    def cal_angle(self, t, hid_idx):
        return t / np.power(self.max_len, hid_idx / self.d_model)

    def get_position_angle_vec(self, t):
        return [self.cal_angle(t, hid_j) for hid_j in range(self.d_model)]

    def forward(self, times):
        times = times.detach().cpu().numpy()
        sinusoid_table = np.array([self.get_position_angle_vec(t) for t in times])
        sinusoid_table[:, 0::2, :] = np.sin(sinusoid_table[:, 0::2, :])  # dim 2i
        sinusoid_table[:, 1::2, :] = np.cos(sinusoid_table[:, 1::2, :])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).permute(0, 2, 1)




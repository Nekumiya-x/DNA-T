import copy
import torch
import torch.nn.functional as F
from torch import nn

from layers.Attention import DNALayer, FullAttentionLayer


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DNAEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, d_ff=256, e_layers=2, max_len=200,DNA_size=5):
        super(DNAEncoder, self).__init__()

        self.layers = get_clones(DNAEncoderLayer(d_model, n_heads, d_ff, max_len,DNA_size=DNA_size), e_layers)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, auxiliary_info=None, dna_mask=None, attn_mask=None):
        # x [B, L, D]
        for layer in self.layers:
            x = layer(x, auxiliary_info, dna_mask=dna_mask, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)

        return x


# DNA-Encoder
class DNAEncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, d_ff=256, max_len=200, dropout=0.05, activation="relu",DNA_size=5):
        super(DNAEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # DeformableNeighborhoodAttention
        self.dna = DNALayer(d_model, n_heads, DNA_size)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_att = FullAttentionLayer(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, auxiliary_info=None, dna_mask=None, attn_mask=None):

        z = self.dna(x, x, x, auxiliary_info=auxiliary_info, attn_mask=dna_mask)
        x = self.norm1(x + self.dropout(z))

        z = self.self_att(x, x, x, attn_mask=attn_mask)
        y = x = self.norm2(x + self.dropout(z))

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

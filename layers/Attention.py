import math
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FullAttentionLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_keys=None,d_values=None):
        super(FullAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)
            scores = scores.masked_fill(attn_mask, -1e4)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class DNALayer(nn.Module):
    def __init__(self, d_model, n_heads,DNA_size, d_keys=None, d_values=None,scale=None, attention_dropout=0.1):
        super(DNALayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.scale = scale
        self.DNA_Size=DNA_size
        self.attndropout = nn.Dropout(attention_dropout)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.offset = nn.Sequential(nn.Linear(d_keys, 2 * d_keys),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.2),
                                         nn.Linear(2 * d_keys, d_keys),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.2),
                                         nn.Linear(d_keys, 1, bias=False))

    @torch.no_grad()
    def NeighborhoodWindows(self, L):
        L = int(L)
        KS = int(self.DNA_Size)
        NS = int(KS / 2)
        A = np.zeros((L, KS), dtype=np.int64)
        for index in range(L):
            left = max(index - NS, 0) + (index + NS >= L) * (L - index - NS - 1)
            r = NS + (index < NS) * (NS - index) + (index + NS >= L) * (L - index - 1 - NS)
            right = index + r
            A[index] = np.arange(left, right + 1)
        return A

    def forward(self, queries, keys, values, auxiliary_info=None,attn_mask=None):

        B, L, _ = queries.shape
        H = self.n_heads

        index_sample = torch.tensor(self.NeighborhoodWindows(L))
        auxiliary_info = auxiliary_info.view(B, L, H, -1).transpose(2, 1)
        auxiliary_info = einops.rearrange(auxiliary_info, 'b h l d -> (b h) l d')
        auxiliary_info = auxiliary_info.unsqueeze(1).expand(-1, L, -1, -1)[:, torch.arange(L).unsqueeze(1),
                         index_sample, :]
        offset = self.offset(auxiliary_info)
        offset = einops.rearrange(offset, '(b h) l k d -> b h l k d', b=B, h=H)
        index_sample = index_sample.unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(offset.device)
        pos = (index_sample + offset).to(queries.dtype)

        _, _, _, K, _ = pos.shape
        pos = einops.rearrange(pos, 'b h l k d -> (b h l) k d')
        pos_grid = pos.unsqueeze(1).expand(B * H * L, 1, K, 2)
        pos_grid = pos_grid / (L - 1) * 2 - 1
        pos_grid[..., 1] = 0.

        keys = keys.unsqueeze(1).expand(-1, L, -1, -1).view(B, L, L, H, -1).permute(0, 3, 1, 4, 2).unsqueeze(-2)
        values = values.unsqueeze(1).expand(-1, L, -1, -1).view(B, L, L, H, -1).permute(0, 3, 1, 4, 2).unsqueeze(-2)
        keys = einops.rearrange(keys, 'b h l d i s -> (b h l) d i s')
        values = einops.rearrange(values, 'b h l d i s -> (b h l) d i s')
        keys = keys.contiguous()
        values = values.contiguous()
        pos_grid=pos_grid.contiguous()

        keys = F.grid_sample(keys, pos_grid, mode='bilinear', align_corners=True).contiguous().squeeze().contiguous().transpose(-1, -2)
        values = F.grid_sample(values, pos_grid, mode='bilinear', align_corners=True).contiguous().squeeze().contiguous().transpose(-1, -2)

        keys = einops.rearrange(keys, '(b h l) k d -> b h l k d', b=B, h=H, l=L).permute(0, 2, 3, 1, 4).contiguous().view(B, L, K, -1)
        values = einops.rearrange(values, '(b h l) k d -> b h l k d', b=B, h=H, l=L).permute(0, 2, 3, 1, 4).contiguous().view(B, L, K, -1)

        queries = self.query_projection(queries).view(B, L, H, -1).permute(0, 2, 1, 3).contiguous().unsqueeze(-2)
        keys = self.key_projection(keys).view(B, L, K, H, -1).permute(0, 3, 1, 2, 4).contiguous()
        values = self.value_projection(values).view(B, L, K, H, -1).permute(0, 3, 1, 2, 4).contiguous()

        B, H, L, K, D = keys.shape
        scale = self.scale or 1. / math.sqrt(D)

        QK = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, H, 1, 1, K)
            QK = QK.masked_fill(attn_mask, -1e4)

        A = self.attndropout(torch.softmax(QK, dim=-1))
        out = torch.matmul(A, values).squeeze(-2).transpose(2, 1).contiguous()

        out = out.view(B, L, -1)

        return self.out_projection(out)


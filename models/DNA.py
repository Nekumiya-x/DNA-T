import torch.nn as nn

from layers.Embed import Embedding
from layers.Encoder import DNAEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.emb = Embedding(c_in=configs.enc_in, d_model=configs.d_model, max_len=configs.max_len)

        self.encoder = DNAEncoder(d_model=configs.d_model,n_heads=configs.n_heads, d_ff=configs.d_ff, e_layers=configs.e_layers, max_len=configs.max_len,DNA_size=configs.DNA_size)

        self.classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.BatchNorm1d(configs.d_model),
            nn.Tanh(),
            nn.Linear(configs.d_model, 2),
        )


    def forward(self, observations, masks, times, deltas,attn_mask):
        # attn_mask
        dna_mask=attn_mask
        batch_size, seq_len, var_num = observations.size()
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask_c = attn_mask.expand(batch_size, seq_len, seq_len)
        attn_mask_r = attn_mask_c.transpose(-2, -1)
        attn_mask= attn_mask_r | attn_mask_c

        # Input embedding
        x, auxiliary_info = self.emb(observations, masks, deltas, times)

        # Encoder
        feat = self.encoder(x, auxiliary_info=auxiliary_info, dna_mask=dna_mask, attn_mask=attn_mask)
        feat = feat.mean(dim=1)
        y = self.classifier(feat)

        return y








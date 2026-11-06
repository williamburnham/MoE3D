import torch
import torch.nn as nn
import math

def sinusoidal(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)                                       # [L, 1]
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d/2]
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0) 


class Transformer(nn.Module):
    def __init__(self, in_dim=263, d_model=64, n_heads=4, n_layers=2,
                 n_classes=2, max_len=196, dropout=0.1):

        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)                         # per-frame projection
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))             # learned summary token
        self.register_buffer("pos", sinusoidal(max_len + 1, d_model))   # +1 for the CLS token
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, pad_mask):

        # x: [B, T, 263], pad_mask: [B, T] (True = padding)
        B = x.shape[0]
        h = self.embed(x)                                               # [B, T, d]
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)           # [B, T+1, d]
        h = h + self.pos[:, : h.shape[1]]
        mask = torch.cat([pad_mask.new_zeros(B, 1), pad_mask], dim=1)   # CLS is never padding
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.head(self.norm(h[:, 0]))                            # logits from CLS
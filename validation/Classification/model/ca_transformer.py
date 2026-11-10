import torch
import torch.nn as nn

from model.transformer import sinusoidal


class CrossModalLayer(nn.Module):
    """One layer for two token streams (motion, EMG), each pre-norm with residual skips:

        x = x + SelfAttn(LN(x))                     within its own modality
        x = x + CrossAttn(LN(x), LN(other))         queries: own stream; keys/values: other stream
        x = x + FFN(LN(x))

    Both cross-attentions read the other stream's post-self-attention state, so the two
    streams are updated symmetrically (neither one goes first)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        def attn():
            return nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        def ffn():
            return nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))

        self.sa = nn.ModuleList([attn(), attn()])            # [motion, emg]
        self.ca = nn.ModuleList([attn(), attn()])            # motion <- emg, emg <- motion
        self.ff = nn.ModuleList([ffn(), ffn()])
        self.sa_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.ca_norm_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.ca_norm_kv = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

    def forward(self, xs, masks):
        # xs: [motion [B, Tm, d], emg [B, Te, d]]; masks: matching [B, T] (True = padding)
        xs = list(xs)
        for i in range(2):                                   # self-attention
            h = self.sa_norm[i](xs[i])
            xs[i] = xs[i] + self.drop(self.sa[i](h, h, h, key_padding_mask=masks[i], need_weights=False)[0])

        ctx = [self.ca_norm_kv[i](xs[1 - i]) for i in range(2)]   # other stream, as it is now
        for i in range(2):                                   # cross-attention
            q = self.ca_norm_q[i](xs[i])
            xs[i] = xs[i] + self.drop(self.ca[i](q, ctx[i], ctx[i], key_padding_mask=masks[1 - i],
                                                 need_weights=False)[0])

        for i in range(2):                                   # feed-forward
            xs[i] = xs[i] + self.drop(self.ff[i](self.ff_norm[i](xs[i])))
        return xs


class CA_Transformer(nn.Module):
    """Two-stream version of Transformer: motion and EMG are embedded separately, each gets a
    CLS token and the same sinusoidal positions (token t = 20 fps frame t in both), then
    n_layers of CrossModalLayer exchange information between them. The two CLS outputs are
    concatenated and classified."""

    def __init__(self, in_dim_motion=263, in_dim_emg=100, d_model=64, n_heads=4, n_layers=2,
                 n_classes=2, max_len=196, dropout=0.1):

        super().__init__()
        self.embed = nn.ModuleList([nn.Linear(in_dim_motion, d_model),     # per-frame projections
                                    nn.Linear(in_dim_emg, d_model)])
        self.cls = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(2)])
        self.register_buffer("pos", sinusoidal(max_len + 1, d_model))     # +1 for the CLS token
        self.layers = nn.ModuleList([CrossModalLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.head = nn.Linear(2 * d_model, n_classes)

    def forward(self, motion, emg, motion_mask, emg_mask):

        # motion: [B, Tm, 263], emg: [B, Te, 100], masks: [B, T] (True = padding)
        B = motion.shape[0]
        xs, masks = [], []
        for i, (x, m) in enumerate([(motion, motion_mask), (emg, emg_mask)]):
            h = self.embed[i](x)                                              # [B, T, d]
            h = torch.cat([self.cls[i].expand(B, -1, -1), h], dim=1)          # [B, T+1, d]
            xs.append(h + self.pos[:, : h.shape[1]])
            masks.append(torch.cat([m.new_zeros(B, 1), m], dim=1))            # CLS is never padding

        for layer in self.layers:
            xs = layer(xs, masks)

        cls = torch.cat([self.norm[i](xs[i][:, 0]) for i in range(2)], dim=-1)   # [B, 2d]
        return self.head(cls)                                                 # logits from both CLS

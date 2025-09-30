# neighbors/head.py
import torch, torch.nn as nn, torch.nn.functional as F

class RelativeEmbedding(nn.Module):
    def __init__(self, bins_range=60, bins_bearing=72, bins_dSOG=32, bins_dCOG=72, d_tok=64):
        super().__init__()
        self.emb_r  = nn.Embedding(bins_range,   d_tok)
        self.emb_th = nn.Embedding(bins_bearing, d_tok)
        self.emb_dv = nn.Embedding(bins_dSOG,    d_tok)
        self.emb_dc = nn.Embedding(bins_dCOG,    d_tok)

    def forward(self, r_idx, th_idx, dv_idx, dc_idx):   # [B,K,T] ints
        return ( self.emb_r(r_idx) + self.emb_th(th_idx)
               + self.emb_dv(dv_idx) + self.emb_dc(dc_idx) )  # [B,K,T,D_tok]

class NeighborGRU(nn.Module):
    def __init__(self, d_tok=64, d_model=256):
        super().__init__()
        self.gru = nn.GRU(d_tok, d_model, num_layers=1, batch_first=True)
    def forward(self, tok):                               # [B,K,T,D_tok]
        B,K,T,D = tok.shape
        _, h = self.gru(tok.reshape(B*K, T, D))
        return h[-1].reshape(B, K, -1)                   # [B,K,D_model]

class NeighborPredictor(nn.Module):
    def __init__(self, bins, d_tok=64, d_model=256, fusion="dot"):
        super().__init__()
        self.embed = RelativeEmbedding(**bins, d_tok=d_tok)
        self.enc   = NeighborGRU(d_tok=d_tok, d_model=d_model)
        self.bias  = nn.Parameter(torch.zeros(1))
        self.fusion = fusion
        #if fusion == "bilinear":
        #    self.W = nn.Parameter(torch.empty(d_model, d_model))
        #    nn.init.xavier_uniform_(self.W)

    def forward(self, h_H, r_idx, th_idx, dv_idx, dc_idx):  # h_H: [B,D]
        tok = self.embed(r_idx, th_idx, dv_idx, dc_idx)      # [B,K,T,D_tok]
        G   = self.enc(tok)                                  # [B,K,D]
        S = (h_H.unsqueeze(1) * G).sum(-1)               # [B,K]
        logits = S + self.bias
        return logits, torch.sigmoid(logits)
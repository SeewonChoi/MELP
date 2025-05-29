import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ST_ECGFormer(nn.Module):
    def __init__(self, seq_len=252, window_size=96, num_leads=12, num_seconds=10, num_layers=12, num_heads=8, embed_dim=128, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.num_leads = num_leads
        self.num_seconds = num_seconds
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # self embed: (B, L, T) -> (B, L, D) T=window_size=96, D=embed_dim=128
        self.embed = nn.Conv1d(in_channels=window_size, out_channels=embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + num_leads, embed_dim), requires_grad=True)
        self.time_embed_set = [nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True) for i in range(num_seconds+1)]
        self.sqatial_embed_set = [nn.Parameter(torch.randn(1, 1 , embed_dim), requires_grad=True) for i in range(num_leads+1)]
        self.cls_token = nn.Parameter(torch.randn(1, num_leads, embed_dim),requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
    
    def forward(self, x, t_indices, sqatial_indices):
        # x: (B, L, T) T=96 -> (B, L, D) D=128
        B, L, T = x.shape
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        # x: (B, L, D) -> (B, L, D) + (1, L, D) add cls token
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        time_embed  = torch.zeros(B, L, self.embed_dim, device=x.device)
        sqatial_embed = torch.zeros(B, L, self.embed_dim, device=x.device)
        for i in range(B):
            for j in range(L):
                try:
                    time_embed[i, j, :] = self.time_embed_set[t_indices[i][j]]
                except:
                    print(i, j)
                    import ipdb
                    ipdb.set_trace()
                sqatial_embed[i, j, :] = self.sqatial_embed_set[sqatial_indices[i][j]]
        x[:,self.num_leads:,:] += time_embed + sqatial_embed
        x = self.dropout(x)
        x = self.encoder(x)
        return x    


class ECGDecoder(nn.Module):
    def __init__(self, num_leads=12, num_layers=4, num_heads=8, embed_dim=128, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.num_leads = num_leads
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        B, C, L, T = x.shape
        x = x.reshape(B, C*L, T)
        x = self.decoder(x, x)

        return x
    
class Classifier(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.pred_head = nn.Linear(embed_dim, 8196)
    
    def forward(self, x):
        x = self.pred_head(x)
        return x
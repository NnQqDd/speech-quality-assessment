import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
# from noresqa import base_encoder, TemporalConvNet


# class NORESQAVoiceClassifier(nn.Module):
#     def __init__(self, num_classes, embed_dim=64):
#         super().__init__()
#         self.base_encoder = base_encoder()
#         self.base_encoder_2 = TemporalConvNet(num_inputs=128,num_channels=[32,64,128,256,embed_dim],kernel_size=3)
#         self.embed_dim = embed_dim
#         self.num_classes = num_classes
#         self.cls_head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x, return_embed=False):
#         """
#         x: [B, 2, T, F]
#         returns:
#             cls: [B, num_classes]
#             embedding: [B, embed_dim]
#         """
#         x = self.base_encoder(x)       # [B, 64, T, 2]
#         x = self.base_encoder_2(x)     # [B, 64, T]
#         embedding = x.mean(dim=-1)
        
#         if not return_embed:
#             return self.cls_head(embedding)
#         return embedding, self.cls_head(embedding)


class RawWaveClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=64, in_channels=1, dropout=0.25, cls_bias=True, cls_norm=False):
        super().__init__()
        # simple conv stack with progressive downsampling
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # pool to fixed-size vector regardless of time length
        self.pool = nn.AdaptiveAvgPool1d(1)  # output shape [B, embed_dim, 1]

        # optional small projection and dropout
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=cls_bias)
        self.cls_norm = cls_norm

    def forward(self, x, return_embed=False):
        """
        x: [B, T] or [B, 1, T]
        returns:
            if return_embed is False: logits [B, num_classes]
            if return_embed is True: (embedding [B, embed_dim], logits [B, num_classes])
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.frontend(x)
        x = self.pool(x).squeeze(-1)

        if not self.cls_norm:
            embeddings = self.proj(x)
            logits = self.cls_head(embeddings)
        else:
            embeddings = self.proj(x)
            E_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            W_norm = torch.nn.functional.normalize(self.cls_head.weight, p=2, dim=1)
            embeddings = E_norm
            logits = torch.nn.functional.linear(E_norm, W_norm) 
            
        if not return_embed:
            return logits
        return embeddings, logits


# def extract_stft(audio, sample_rate = 16000):
#     fx, tx, stft_out = signal.stft(audio, sample_rate, window='hann',nperseg=512,noverlap=256,nfft=512)
#     stft_out = stft_out[:256,:]
#     feat = np.concatenate((np.abs(stft_out).reshape([stft_out.shape[0],stft_out.shape[1],1]), np.angle(stft_out).reshape([stft_out.shape[0],stft_out.shape[1],1])), axis=2)
#     return feat
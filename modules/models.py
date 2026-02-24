import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2Model, WavLMModel


class EncoderClassifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str,
        layer: int | None = None,
        embed_dim: int = 64,
        cls_bias=True, cls_norm=False
    ):
        super().__init__()
        if 'wavlm' in model_name:
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif 'hubert' in model_name:
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        elif 'wav2vec2' in model_name:
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        else:
            raise ValueError("What?")
        self.layer = layer
        self.cls_head = nn.Linear(self.model.config.hidden_size, num_classes, bias=cls_bias)
        self.cls_norm = cls_norm

    def forward(self, waveform, return_embed=False):
        outputs = self.model(
            waveform,
            output_hidden_states=self.layer is not None,
        )
        if self.layer is not None:
            hidden = outputs.hidden_states[self.layer]
        else:
            hidden = outputs.last_hidden_state
        embeddings = hidden.mean(dim=1)
        if not self.cls_norm:
            logits = self.cls_head(embeddings)
        else:
            E_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            W_norm = torch.nn.functional.normalize(self.cls_head.weight, p=2, dim=1)
            embeddings = E_norm
            logits = torch.nn.functional.linear(E_norm, W_norm) 
            
        if not return_embed:
            return logits
        return embeddings, logits


class RawWaveClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=64, in_channels=1, dropout=0.1, cls_bias=True, cls_norm=False):
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.frontend(x)
        x = self.pool(x).squeeze(-1)
        embeddings = self.proj(x)

        if not self.cls_norm:
            logits = self.cls_head(embeddings)
        else:
            E_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            W_norm = torch.nn.functional.normalize(self.cls_head.weight, p=2, dim=1)
            embeddings = E_norm
            logits = torch.nn.functional.linear(E_norm, W_norm) 
            
        if not return_embed:
            return logits
        return embeddings, logits


if __name__ == "__main__":
    import librosa
    wave, _ = librosa.load('/home/duyn/ActableDuy/voice-synthesis/reference_audio.wav', mono=True, sr=16000)
    device = torch.device('cuda')
    waves = torch.from_numpy(wave).unsqueeze(0).to(device)
    
    print("----------------HuBERT----------------")
    model = EncoderClassifier(num_classes=2, model_name='wav2vec2')
    model.to(device)
    model.eval()
    with torch.no_grad():
        emb, out = model(waves, return_embed=True)
        print(emb.shape, out.shape)

    print("----------------WAV2VEC2----------------")
    model = EncoderClassifier(num_classes=2, model_name='hubert')
    model.to(device)
    model.eval()
    with torch.no_grad():
        emb, out = model(waves, return_embed=True)
        print(emb.shape, out.shape)

    print("----------------WAVLM----------------")
    model = EncoderClassifier(num_classes=2, model_name='wavlm')
    model.to(device)
    model.eval()
    with torch.no_grad():
        emb, out = model(waves, return_embed=True)
        print(emb.shape, out.shape)

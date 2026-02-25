import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from transformers import HubertModel, Wav2Vec2Model, WavLMModel
try:
    from .wespeaker.model import ResNet34
    from .ecapa_tdnn.model import ECAPA_TDNN
except:
    from wespeaker.model import ResNet34
    from ecapa_tdnn.model import ECAPA_TDNN


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


class Wespeaker34(torch.nn.Module):
    def __init__(self, num_classes, embed_dim=256, cls_bias=True, cls_norm=False):
        super().__init__()
        self.model = ResNet34(feat_dim=80, embed_dim=256, pooling_func='TSTP', two_emb_layer=False)
        self.model.seg_1 = torch.nn.Linear(in_features=5120, out_features=embed_dim)
        self.cls_head = torch.nn.Linear(embed_dim, num_classes, bias=cls_bias)
        self.cls_norm = cls_norm 
        
    def compute_fbank(self,
                      waveforms,
                      sample_rate=16000,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      cmn=True):
        B = waveforms.shape[0]
        feats = []
        for i in range(B):
            feat = kaldi.fbank(waveforms[i].unsqueeze(0),
                               num_mel_bins=num_mel_bins,
                               frame_length=frame_length,
                               frame_shift=frame_shift,
                               sample_frequency=sample_rate,
                               window_type='hamming')
            if cmn:
                feat = feat - torch.mean(feat, 0)
            feats.append(feat)
        return torch.stack(feats)
    
    def forward(self, pcm, return_embed=False):
        pcm = torch.clamp(pcm, -1.0, 1.0)
        pcm = (pcm * 32767)
        feats = self.compute_fbank(pcm, sample_rate=16000, cmn=True)
        outputs = self.model(feats)
        embeddings = outputs[-1]
        
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
    


class ECAPA(torch.nn.Module):
    def __init__(self, num_classes, embed_dim=192, cls_bias=True, cls_norm=False):
        super().__init__()
        self.model = ECAPA_TDNN()
        self.model.fc6 = torch.nn.Linear(in_features=3072, out_features=embed_dim)
        self.model.bn6 = torch.nn.BatchNorm1d(embed_dim)
        self.cls_head = torch.nn.Linear(embed_dim, num_classes, bias=cls_bias)
        self.cls_norm = cls_norm 


    def forward(self, waves, return_embed=False):
        embeddings = self.model(waves)
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

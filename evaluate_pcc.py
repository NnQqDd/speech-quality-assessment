import os
import pandas as pd
import librosa
from tqdm import tqdm
import torch
import numpy as np
# from torchmetrics.functional import pearson_corrcoef
from modules.models import *


DATASET_PATH = '/home/duyn/ActableDuy/datasets/NISQA_Corpus'
METADATA_PATH = os.path.join(DATASET_PATH, 'NISQA_corpus_file.csv')
DEVICE = torch.device("cuda")
MODEL = RawWaveClassifier(num_classes=16)
# MODEL = RawWaveClassifier(num_classes=16, cls_bias=False, cls_norm=True)
MODEL.load_state_dict(torch.load("/home/duyn/ActableDuy/speech-quality-assessment/weights/t6fko2em/best.pth"))
# MODEL.load_state_dict(torch.load("/home/duyn/ActableDuy/speech-quality-assessment/weights/f7qddp8m/best.pth"))
MODEL.eval()
MODEL.to(DEVICE)
NUM_TRAIN_ROWS = None
NUM_TEST_ROWS = None
MAE = 0
MSE = 0

metadata_df = pd.read_csv(METADATA_PATH)
print(metadata_df.nunique())

queries = []
embeddings = []
for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
    if 'VAL' in row['db']:
        continue

    if 'TRAIN' in row['db']:
        if NUM_TRAIN_ROWS is not None:
            NUM_TRAIN_ROWS -= 1
        if NUM_TRAIN_ROWS is not None and NUM_TRAIN_ROWS < 0:
            continue
        filepath = os.path.join(DATASET_PATH, row['filepath_deg'])
        wave, _ = librosa.load(filepath, mono=True, sr=16000)
        wave = torch.from_numpy(wave).to(DEVICE)
        wave = wave.unsqueeze(0)
        with torch.no_grad():
            emb, _ = MODEL(wave, return_embed=True)
        embeddings.append((emb[0], row['mos']))
    if 'TEST' in row['db']:
        if NUM_TEST_ROWS is not None:
            NUM_TEST_ROWS -= 1
        if NUM_TEST_ROWS is not None and NUM_TEST_ROWS < 0:
            continue
        filepath = os.path.join(DATASET_PATH, row['filepath_deg'])
        wave, _ = librosa.load(filepath, mono=True, sr=16000)
        wave = torch.from_numpy(wave).to(DEVICE)
        wave = wave.unsqueeze(0)
        with torch.no_grad():
            emb, _ = MODEL(wave, return_embed=True)
        queries.append((emb[0], row['mos']))

        
X_db = torch.stack([e for e, _ in embeddings])  # (N, D)
mos_db = torch.tensor([m for _, m in embeddings]).to(DEVICE)             # (N,)
X_q = torch.stack([e for e, _ in queries]) 
mos_q = torch.tensor([m for _, m in queries]).to(DEVICE)

X_q_c  = X_q  - X_q.mean(dim=1, keepdim=True)
X_db_c = X_db - X_db.mean(dim=1, keepdim=True)
X_q_n  = X_q_c  / torch.linalg.norm(X_q_c,  dim=1, keepdim=True).clamp_min(1e-9)
X_db_n = X_db_c / torch.linalg.norm(X_db_c, dim=1, keepdim=True).clamp_min(1e-9)
sim = X_q_n @ X_db_n.T

print("SHAPES    :", X_db.shape, X_q.shape)
print("TARGET MOS:", mos_db.max().item(), mos_db.min().item(), mos_db.mean().item(), mos_db.std().item())
print("PRED MOS  :", mos_q.max().item(), mos_q.min().item(), mos_q.mean().item(), mos_q.std().item())
for K in range(8, 65):
    _, topk_idx = torch.topk(sim, K, dim=1)
    avg_mos = mos_db[topk_idx]
    avg_mos = avg_mos.mean(axis=1)
    print(f"K={K}, MAE", torch.abs(avg_mos - mos_q).mean().item())
    print(f"K={K}, MSE", torch.abs((avg_mos - mos_q)**2).mean().item())
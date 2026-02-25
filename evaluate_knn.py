import warnings; warnings.filterwarnings("ignore")
import os
import yaml
import pandas as pd
import librosa
from tqdm import tqdm
import torch
import numpy as np
from scipy.stats import pearsonr
from modules.models import *
from modules.utilities import *


DATASET_PATH = '/home/duyn/ActableDuy/datasets/NISQA_Corpus'
METADATA_PATH = os.path.join(DATASET_PATH, 'NISQA_corpus_file.csv')
WEIGHT_PATH = "/home/duyn/ActableDuy/speech-quality-assessment/weights/8dpjzh8a/best.pth"
CONFIG_PATH = "/home/duyn/ActableDuy/speech-quality-assessment/weights/8dpjzh8a/config.yaml"
DEVICE = torch.device("cuda")
NUM_TRAIN_ROWS = None
NUM_TEST_ROWS = None
MAE = 0
MSE = 0

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
ModelClass = load_class(config['model']['name'])
if 'args' in config['model'] and config['model'] is not None:
    config['model']['args']['num_classes'] = 16
    MODEL = ModelClass(**config['model']['args'])
else:
    MODEL = ModelClass(num_classes=16)
assert isinstance(MODEL, torch.nn.Module)
MODEL.to(DEVICE)
MODEL.eval()

print(WEIGHT_PATH)
print(CONFIG_PATH)
print(DATASET_PATH)
print(NUM_TRAIN_ROWS)
print(NUM_TEST_ROWS)

metadata_df = pd.read_csv(METADATA_PATH)
# print(metadata_df.nunique())

queries = []
database = []
for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
    if 'VAL' in row['db']:
        continue
    if 'TRAIN' in row['db'] and NUM_TRAIN_ROWS is not None:
        NUM_TRAIN_ROWS -= 1
    if 'TRAIN' in row['db'] and NUM_TRAIN_ROWS is not None and NUM_TRAIN_ROWS < 0:
        continue
    if 'TEST' in row['db'] and NUM_TEST_ROWS is not None:
        NUM_TEST_ROWS -= 1
    if 'TEST' in row['db'] and NUM_TEST_ROWS is not None and NUM_TEST_ROWS < 0:
        continue
    filepath = os.path.join(DATASET_PATH, row['filepath_deg'])
    filepath = os.path.join(DATASET_PATH, row['filepath_deg'])
    wave, _ = librosa.load(filepath, mono=True, sr=16000)
    wave = torch.from_numpy(wave).to(DEVICE)
    waves = wave.unsqueeze(0)
    with torch.no_grad():
        emb, _ = MODEL(waves, return_embed=True)
    if 'TRAIN' in row['db']:
        database.append((emb[0], row['mos']))
    if 'TEST' in row['db']:
        queries.append((emb[0], row['mos']))
        
X_db = torch.stack([e for e, _ in database])  # (N, D)
mos_db = torch.tensor([m for _, m in database]).to(DEVICE)             # (N,)
# queries
X_q = torch.stack([e for e, _ in queries]) 
mos_q = torch.tensor([m for _, m in queries]).to(DEVICE)
# normalize
X_db = F.normalize(X_db, dim=1)
X_q  = F.normalize(X_q,  dim=1)
# cosine similarity
sim = X_q @ X_db.T  # (Q, N)


print("SHAPES    :", tuple(X_db.shape), tuple(X_q.shape))
print("TARGET MOS:", mos_db.max().item(), mos_db.min().item(), mos_db.mean().item(), mos_db.std().item())
print("PRED MOS  :", mos_q.max().item(), mos_q.min().item(), mos_q.mean().item(), mos_q.std().item())
for K in range(1, 65):
    _, topk_idx = torch.topk(sim, K, dim=1)
    avg_mos = mos_db[topk_idx].mean(dim=1)
    print(f"K={K}, MAE", torch.abs(avg_mos - mos_q).mean().item())
    print(f"K={K}, MSE", torch.abs((avg_mos - mos_q)**2).mean().item())
    pcc, p_value = pearsonr(avg_mos.cpu().numpy(), mos_q.cpu().numpy())
    print(f"K={K}, PEAR", pcc, p_value)
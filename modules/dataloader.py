from typing import *
from collections import defaultdict
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(
            self, 
            audio_metadatas: List[Dict],
            sample_rate: int=16000,
            ):
        self.audio_metadatas = audio_metadatas
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.audio_metadatas)

    def __getitem__(self, idx):
        audio_metadata = self.audio_metadatas[idx]
        audio_path = audio_metadata['filepath']
        label = audio_metadata['label']
        # label = torch.tensor(label, dtype=torch.long)
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return waveform, label


def collate(batch):
    waveforms, labels = zip(*batch)
    
    waveforms = list(waveforms)
    labels = list(labels)
    if isinstance(labels[0], int):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.float)
    
    max_length = 0
    for waveform in waveforms:
        max_length = max(max_length, len(waveform))
    for i in range(len(waveforms)):
        waveforms[i] = np.pad(waveforms[i], (0, max_length - waveforms[i].shape[0]), mode='constant')
        waveforms[i] = torch.from_numpy(waveforms[i])
    waveforms = torch.stack(waveforms, dim=0)

    return waveforms, labels
    

def prepare_dataloaders(ds_config, dl_config):
    datasets = dict()
    audio_metadatas = pd.read_csv(ds_config['audio_metadata']).to_dict(orient='records')
    
    train_audio_metadatas = [meta for meta in audio_metadatas if meta['split'] == 'train']
    valid_audio_metadatas = [meta for meta in audio_metadatas if meta['split'] == 'valid']
    num_classes = len(set([meta['label'] for meta in train_audio_metadatas]))
    datasets['train'] = MyDataset(
        audio_metadatas=train_audio_metadatas,
        sample_rate=ds_config['sample_rate'],
    )
    datasets['valid'] = MyDataset(
        audio_metadatas=valid_audio_metadatas,
        sample_rate=ds_config['sample_rate'],
    )

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        datasets['train'],
        collate_fn=collate,
        **dl_config['train'] 
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        collate_fn=collate,
        **dl_config['valid'] 
    )

    return dataloaders, num_classes
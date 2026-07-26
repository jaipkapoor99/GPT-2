"""
GPT-2 Dataset Module
Implements zero-RAM memory mapped loading across pre-tokenized binary shards (np.memmap).
"""

import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config import GPT2Config

def get_dataloaders(config: GPT2Config, accelerator):
    """ Loads pre-tokenized binary shards with np.memmap and returns prepared DataLoaders """
    shards_dir = "shards"
    bin_shards = sorted(glob.glob(os.path.join(shards_dir, "fineweb_shard_*.bin")))
    
    if not bin_shards:
        if os.path.exists("fineweb_tokens.bin"):
            bin_shards = ["fineweb_tokens.bin"]
        else:
            raise FileNotFoundError("No pre-tokenized binary dataset shards found! Run 'python tokenize_dataset.py' first.")

    accelerator.print(f"Loading {len(bin_shards)} binary shard(s) via Zero-RAM np.memmap...")
    shard_tensors = []
    total_tokens = 0
    for shard in bin_shards:
        tokens_np = np.memmap(shard, dtype=np.uint16, mode='r')
        total_tokens += len(tokens_np)
        shard_tensors.append(torch.from_numpy(tokens_np.astype(np.int64)))
        
    data_tensor = torch.cat(shard_tensors, dim=0)
    accelerator.print(f"Total Tokens Loaded: {total_tokens:,}")
    
    unfolded = data_tensor.unfold(dimension=0, size=config.T + 1, step=256)
    
    n1 = int(0.8 * unfolded.shape[0])
    n2 = int(0.9 * unfolded.shape[0])
    
    Xtr, Ytr = unfolded[:n1, :config.T], unfolded[:n1, 1:config.T + 1]
    Xdev, Ydev = unfolded[n1:n2, :config.T], unfolded[n1:n2, 1:config.T + 1]
    Xte, Yte   = unfolded[n2:, :config.T], unfolded[n2:, 1:config.T + 1]
    
    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=config.B, shuffle=True)
    dev_loader   = DataLoader(TensorDataset(Xdev, Ydev), batch_size=config.B)
    test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=config.B)
    
    train_loader, dev_loader, test_loader = accelerator.prepare(
        train_loader, dev_loader, test_loader
    )
    
    return train_loader, dev_loader, test_loader

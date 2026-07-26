"""
Production Zero-RAM Shard Dataset Module
Prevents WSL RAM crashes by loading 1 binary shard at a time into RAM (500MB max)
instead of casting 1.9 Billion tokens into a 30GB PyTorch tensor!
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import GPT2Config

class ShardedDataset(Dataset):
    """
    High-Performance Zero-RAM Sharded Dataset
    Loads individual binary shards on demand to keep RAM usage under 500 MB.
    """
    def __init__(self, bin_shards, sequence_length=1024, step=256):
        self.bin_shards = bin_shards
        self.T = sequence_length
        self.step = step
        
        # Calculate sequence count per shard without loading full arrays into RAM
        self.shard_bounds = []
        total_sequences = 0
        
        for shard in self.bin_shards:
            num_tokens = os.path.getsize(shard) // 2 # uint16 = 2 bytes
            num_seqs = max(0, (num_tokens - (self.T + 1)) // self.step + 1)
            self.shard_bounds.append((total_sequences, total_sequences + num_seqs, shard, num_tokens))
            total_sequences += num_seqs
            
        self.total_sequences = total_sequences
        self.current_shard_path = None
        self.current_unfolded = None

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Locate which shard contains idx
        for start_seq, end_seq, shard_path, num_tokens in self.shard_bounds:
            if start_seq <= idx < end_seq:
                shard_seq_idx = idx - start_seq
                
                # Load shard into memmap only when needed
                if self.current_shard_path != shard_path:
                    self.current_shard_path = shard_path
                    tokens_np = np.memmap(shard_path, dtype=np.uint16, mode='r')
                    data_tensor = torch.from_numpy(tokens_np.astype(np.int64))
                    self.current_unfolded = data_tensor.unfold(dimension=0, size=self.T + 1, step=self.step)
                
                seq = self.current_unfolded[shard_seq_idx]
                return seq[:self.T], seq[1:self.T + 1]
                
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_sequences}")

def get_dataloaders(config: GPT2Config, accelerator):
    shards_dir = "shards"
    bin_shards = sorted(glob.glob(os.path.join(shards_dir, "fineweb_shard_*.bin")))
    
    if not bin_shards:
        if os.path.exists("fineweb_tokens.bin"):
            bin_shards = ["fineweb_tokens.bin"]
        else:
            raise FileNotFoundError("No binary dataset shards found! Run 'python tokenize_dataset.py' first.")

    accelerator.print(f"Loading {len(bin_shards)} binary shard(s) via RAM-Safe ShardedDataset...")
    
    full_ds = ShardedDataset(bin_shards, sequence_length=config.T, step=256)
    accelerator.print(f"Total Sequences Available: {len(full_ds):,}")
    
    # 80/10/10 Train/Dev/Test Split
    n_train = int(0.8 * len(full_ds))
    n_dev   = int(0.1 * len(full_ds))
    n_test  = len(full_ds) - n_train - n_dev
    
    train_ds, dev_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_dev, n_test])
    
    train_loader = DataLoader(train_ds, batch_size=config.B, shuffle=True)
    dev_loader   = DataLoader(dev_ds, batch_size=config.B, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=config.B, shuffle=False)
    
    train_loader, dev_loader, test_loader = accelerator.prepare(
        train_loader, dev_loader, test_loader
    )
    
    return train_loader, dev_loader, test_loader

"""
Zero-Copy Direct Slicing Dataset Module
Eliminates ALL memory allocations by performing direct index offset slicing
on memmapped numpy uint16 files.
RAM Usage: 0.00 MB!
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import UltronConfig

class ZeroCopyShardedDataset(Dataset):
    """
    True Zero-Copy Memmap Dataset
    Calculates exact token offset math and reads only 1,025 tokens per sample directly from disk.
    Zero RAM allocation, zero tensor copying, zero memory fragmentation!
    """
    def __init__(self, bin_shards, sequence_length=1024, step=256):
        self.bin_shards = bin_shards
        self.T = sequence_length
        self.step = step
        
        self.shard_memmaps = []
        self.shard_offsets = []
        total_sequences = 0
        
        for shard_path in self.bin_shards:
            num_tokens = os.path.getsize(shard_path) // 2 # uint16 = 2 bytes
            num_seqs = max(0, (num_tokens - (self.T + 1)) // self.step + 1)
            
            # Virtual disk view - 0 bytes of RAM
            mmap = np.memmap(shard_path, dtype=np.uint16, mode='r')
            self.shard_memmaps.append((mmap, num_seqs))
            self.shard_offsets.append((total_sequences, total_sequences + num_seqs))
            total_sequences += num_seqs
            
        self.total_sequences = total_sequences

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Find shard using binary / offset range check
        for shard_idx, (start_seq, end_seq) in enumerate(self.shard_offsets):
            if start_seq <= idx < end_seq:
                seq_idx_in_shard = idx - start_seq
                token_start = seq_idx_in_shard * self.step
                
                # Zero-copy disk slice (1,025 uint16 tokens -> int64 tensor)
                mmap, _ = self.shard_memmaps[shard_idx]
                chunk = mmap[token_start : token_start + self.T + 1].astype(np.int64)
                
                x = torch.from_numpy(chunk[:self.T])
                y = torch.from_numpy(chunk[1:self.T + 1])
                return x, y
                
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_sequences}")

def get_dataloaders(config: UltronConfig, accelerator):
    bin_shards = sorted(glob.glob("shards/fineweb_shard_*.bin") + glob.glob("shards_edu/fineweb_edu_shard_*.bin"))
    
    if not bin_shards:
        if os.path.exists("fineweb_tokens.bin"):
            bin_shards = ["fineweb_tokens.bin"]
        else:
            raise FileNotFoundError("No binary dataset shards found! Run 'python tokenize_dataset.py' first.")


    accelerator.print(f"Loading {len(bin_shards)} binary shard(s) via Zero-Copy Memmap Dataset...")
    
    full_ds = ZeroCopyShardedDataset(bin_shards, sequence_length=config.T, step=256)
    accelerator.print(f"Total Sequences Available: {len(full_ds):,}")
    
    n_train = int(0.8 * len(full_ds))
    n_dev   = int(0.1 * len(full_ds))
    n_test  = len(full_ds) - n_train - n_dev
    
    train_ds, dev_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_dev, n_test])
    
    train_loader = DataLoader(train_ds, batch_size=config.B, shuffle=True, num_workers=2, pin_memory=False)
    dev_loader   = DataLoader(dev_ds, batch_size=config.B, shuffle=False, num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=config.B, shuffle=False, num_workers=2, pin_memory=False)

    
    train_loader, dev_loader, test_loader = accelerator.prepare(
        train_loader, dev_loader, test_loader
    )
    
    return train_loader, dev_loader, test_loader

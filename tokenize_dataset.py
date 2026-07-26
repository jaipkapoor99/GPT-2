import os
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def main(shard_size_tokens=100_000_000, max_shards=100):
    """
    Production-grade script to stream ALL of FineWeb sample-10BT (10 Billion Tokens)
    and save them into compact 100M token binary shards (uint16 format).
    """
    output_dir = "shards"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== FULL FINEWEB 10B TOKEN PRE-TOKENIZATION SCRIPT ===")
    print(f"Target Shard Size : {shard_size_tokens:,} tokens (~200 MB per shard)")
    print(f"Output Directory   : {os.path.abspath(output_dir)}")
    
    print("\nLoading modern compact SmolLM BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    print(f"Vocabulary Size     : {tokenizer.vocab_size:,} tokens")
    
    print("\nStreaming FineWeb (sample-10BT)...")
    raw_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    iterator = iter(raw_dataset)
    
    current_tokens = []
    shard_index = 0
    total_tokens_processed = 0
    chunk_size_docs = 1000
    
    pbar = tqdm(desc="Tokens Processed", unit="tok", unit_scale=True)
    
    while shard_index < max_shards:
        try:
            chunk_docs = [next(iterator)['text'] for _ in range(chunk_size_docs)]
        except StopIteration:
            break
            
        chunk_text = "\n\n".join(chunk_docs)
        chunk_tokens = tokenizer.encode(chunk_text, verbose=False)
        current_tokens.extend(chunk_tokens)
        
        pbar.update(len(chunk_tokens))
        total_tokens_processed += len(chunk_tokens)
        
        # Save shard when buffer reaches 100M tokens
        if len(current_tokens) >= shard_size_tokens:
            shard_tokens = np.array(current_tokens[:shard_size_tokens], dtype=np.uint16)
            current_tokens = current_tokens[shard_size_tokens:] # Keep remainder
            
            shard_filename = os.path.join(output_dir, f"fineweb_shard_{shard_index:04d}.bin")
            shard_tokens.tofile(shard_filename)
            
            meta_filename = os.path.join(output_dir, f"fineweb_shard_{shard_index:04d}_meta.json")
            with open(meta_filename, "w") as f:
                json.dump({
                    "shard_index": shard_index,
                    "tokens": shard_size_tokens,
                    "vocab_size": tokenizer.vocab_size,
                    "dtype": "uint16"
                }, f, indent=2)
                
            pbar.write(f"✓ Saved Shard {shard_index:04d} ({shard_size_tokens:,} tokens -> {shard_filename})")
            shard_index += 1
            
    # Save final partial shard
    if len(current_tokens) > 0 and shard_index < max_shards:
        shard_tokens = np.array(current_tokens, dtype=np.uint16)
        shard_filename = os.path.join(output_dir, f"fineweb_shard_{shard_index:04d}.bin")
        shard_tokens.tofile(shard_filename)
        
        meta_filename = os.path.join(output_dir, f"fineweb_shard_{shard_index:04d}_meta.json")
        with open(meta_filename, "w") as f:
            json.dump({
                "shard_index": shard_index,
                "tokens": len(current_tokens),
                "vocab_size": tokenizer.vocab_size,
                "dtype": "uint16"
            }, f, indent=2)
        pbar.write(f"✓ Saved Final Shard {shard_index:04d} ({len(current_tokens):,} tokens -> {shard_filename})")

    pbar.close()
    print(f"\nPre-tokenization Complete! Total Tokens Processed: {total_tokens_processed:,}")

if __name__ == "__main__":
    main()

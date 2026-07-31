import os
import sys
import json
import glob
import time
import signal
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telemetry import TokenizationTelemetry

def main(shard_size_tokens=100_000_000, max_shards=100):
    """
    Production-grade script to stream FineWeb-Edu sample-10BT (10 Billion Tokens)
    and save them into compact 100M token binary shards (uint16 format).
    Incorporate TokenizationTelemetry and handles KeyboardInterrupt gracefully.
    """
    output_dir = "shards_edu"
    os.makedirs(output_dir, exist_ok=True)
    
    existing_shards = sorted(glob.glob(os.path.join(output_dir, "fineweb_edu_shard_*.bin")))
    start_shard = len(existing_shards)
    target_total_tokens = max_shards * shard_size_tokens
    
    print(f"=== FINEWEB-EDU PRE-TOKENIZATION PIPELINE ===")
    print(f"Existing Shards   : {start_shard} shards ({start_shard * shard_size_tokens / 1e9:.2f}B tokens)")
    print(f"Resuming at Shard : fineweb_edu_shard_{start_shard:04d}.bin")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    eos_token_id = tokenizer.eos_token_id or 0

    raw_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    iterator = iter(raw_dataset)

    # Fast-forward past existing shards by token length approximation
    tokens_to_skip = start_shard * shard_size_tokens
    if tokens_to_skip > 0:
        print(f"Fast-forwarding stream past {tokens_to_skip:,} tokens...")
        skipped = 0
        for doc in iterator:
            approx_toks = len(doc['text']) // 4
            skipped += approx_toks
            if skipped >= tokens_to_skip:
                break

    telemetry = TokenizationTelemetry(target_tokens=target_total_tokens, start_tokens=tokens_to_skip)
    current_tokens = []
    shard_index = start_shard
    total_tokens_processed = tokens_to_skip
    batch_size = 1000

    # Signal handler for clean keyboard interruption
    def handle_interrupt(signum, frame):
        telemetry.print_message(f"\n⚠️ KeyboardInterrupt received. Safely stopping tokenization at {total_tokens_processed / 1e9:.2f}B tokens...")
        os._exit(0)
    
    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        while shard_index < max_shards:
            batch_docs = []
            for _ in range(batch_size):
                try:
                    batch_docs.append(next(iterator)['text'])
                except StopIteration:
                    break

            if not batch_docs:
                break

            # Fast parallel batch encoding via Rust backend
            if hasattr(tokenizer, "backend_tokenizer"):
                encodings = tokenizer.backend_tokenizer.encode_batch(batch_docs, add_special_tokens=False)
                for enc in encodings:
                    tokens = list(enc.ids)
                    tokens.append(eos_token_id)
                    current_tokens.extend(tokens)
                    total_tokens_processed += len(tokens)
                    telemetry.update(added_tokens=len(tokens), current_total=total_tokens_processed)
            else:
                for doc_text in batch_docs:
                    tokens = tokenizer.encode(doc_text, verbose=False)
                    tokens.append(eos_token_id)
                    current_tokens.extend(tokens)
                    total_tokens_processed += len(tokens)
                    telemetry.update(added_tokens=len(tokens), current_total=total_tokens_processed)

            while len(current_tokens) >= shard_size_tokens and shard_index < max_shards:
                shard_tokens = np.array(current_tokens[:shard_size_tokens], dtype=np.uint16)
                current_tokens = current_tokens[shard_size_tokens:]
                
                shard_filename = os.path.join(output_dir, f"fineweb_edu_shard_{shard_index:04d}.bin")
                shard_tokens.tofile(shard_filename)
                
                meta_filename = os.path.join(output_dir, f"fineweb_edu_shard_{shard_index:04d}_meta.json")
                with open(meta_filename, "w") as f:
                    json.dump({
                        "shard_index": shard_index,
                        "tokens": shard_size_tokens,
                        "vocab_size": tokenizer.vocab_size,
                        "dtype": "uint16"
                    }, f, indent=2)
                    
                telemetry.print_message(f"✓ Saved Shard {shard_index:04d} ({shard_size_tokens:,} tokens -> {shard_filename})")
                shard_index += 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
            
    telemetry.close()
    print(f"\n🎉 Pre-tokenization Complete! Total Tokens Processed: {total_tokens_processed:,}")

if __name__ == "__main__":
    main()

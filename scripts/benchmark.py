"""
Benchmark Evaluation Script for GPT-2 using lm-evaluation-harness
"""
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 using lm-eval")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face repo ID")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,wikitext", help="Comma-separated list of tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation examples (for testing)")
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    args = parser.parse_args()

    print(f"Starting evaluation of {args.repo_id} on tasks: {args.tasks}")
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.repo_id},trust_remote_code=True",
        "--tasks", args.tasks,
        "--device", "cuda:0",
        "--batch_size", args.batch_size,
        "--output_path", args.output
    ]
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
        
    cmd_str = " ".join(cmd)
    print(f"Running command:\n{cmd_str}\n")
    
    # Run the evaluation command
    exit_code = os.system(cmd_str)
    
    if exit_code == 0:
        print(f"\nEvaluation completed successfully. Results saved to {args.output}")
    else:
        print(f"\nEvaluation failed with exit code {exit_code}")

if __name__ == "__main__":
    main()

"""
Unit and Integration Tests for Ultron (124M 2026 SOTA)
Using torch.testing assertions, Accelerate device management, rich colorful logs, and emojis.
"""

import sys, os

# ── Accelerate guard ────────────────────────────────────────────────────────
if not any(k in os.environ for k in [
    "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
]):
    raise RuntimeError("Run with: accelerate launch <script>.py ...")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.testing as tt
import unittest
from accelerate import Accelerator
from rich.console import Console
from config import UltronConfig
from model import UltronModel, RMSNorm, apply_rotary_emb

console = Console()
accelerator = Accelerator()
device = accelerator.device

class TestUltronModel(unittest.TestCase):

    def setUp(self):
        console.print(f"[bold cyan]⚙️ Setting up test on device:[/bold cyan] [bold yellow]{device}[/bold yellow]")

    def test_config_defaults(self):
        console.print("[bold blue]📋 Testing UltronConfig default parameters...[/bold blue]")
        cfg = UltronConfig()
        console.print(f"  [green]• C:[/green] [bold white]{cfg.C}[/bold white], [green]n_head:[/green] [bold white]{cfg.n_head}[/bold white], [green]n_kv_head:[/green] [bold white]{cfg.n_kv_head}[/bold white], [green]head_dim:[/green] [bold white]{cfg.head_dim}[/bold white], [green]vocab_size:[/green] [bold white]{cfg.vocab_size}[/bold white]")
        self.assertEqual(cfg.C, 768)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.n_kv_head, 4)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.vocab_size, 49152)
        self.assertEqual(cfg.grad_accum_steps, 4)
        console.print("[bold green]✅ test_config_defaults passed successfully![/bold green]\n")

    def test_model_forward_shape(self):
        console.print("[bold blue]⚡ Testing Ultron model forward pass output shape...[/bold blue]")
        cfg = UltronConfig(n_layer=2)
        model = UltronModel(cfg)
        model = accelerator.prepare(model)
        model.eval()
        
        B, T = 2, 64
        idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long, device=device)
        console.print(f"  [magenta]📥 Input Tensor Shape:[/magenta] [bold white]{idx.shape}[/bold white]")
        
        with torch.no_grad():
            out = model(idx)
            
        console.print(f"  [magenta]📤 Forward Output Logits Shape:[/magenta] [bold white]{out.logits.shape}[/bold white]")
        tt.assert_close(torch.tensor(out.logits.shape), torch.tensor([B, T, cfg.vocab_size]))
        console.print("[bold green]✅ test_model_forward_shape passed successfully![/bold green]\n")

    def test_torch_compile_forward(self):
        console.print("[bold blue]🔥 Testing Ultron model graph compilation with torch.compile()...[/bold blue]")
        cfg = UltronConfig(n_layer=2)
        model = UltronModel(cfg)
        model = accelerator.prepare(model)
        model.eval()
        model = torch.compile(model)
        
        B, T = 2, 64
        idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model(idx)
            
        console.print(f"  [magenta]⚡ Compiled Model Output Shape:[/magenta] [bold white]{out.logits.shape}[/bold white]")
        tt.assert_close(torch.tensor(out.logits.shape), torch.tensor([B, T, cfg.vocab_size]))
        console.print("[bold green]✅ test_torch_compile_forward passed successfully![/bold green]\n")

    def test_model_loss_computation(self):
        console.print("[bold blue]🎯 Testing Ultron model cross-entropy loss computation...[/bold blue]")
        cfg = UltronConfig(n_layer=2)
        model = UltronModel(cfg)
        model = accelerator.prepare(model)
        
        B, T = 2, 64
        idx = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long, device=device)
        targets = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long, device=device)
        
        out = model(idx, targets=targets)
        console.print(f"  [yellow]🔥 Initial Loss Value:[/yellow] [bold cyan]{out.loss.item():.4f}[/bold cyan]")
        self.assertIsNotNone(out.loss)
        self.assertTrue(out.loss.item() > 0.0)
        console.print("[bold green]✅ test_model_loss_computation passed successfully![/bold green]\n")

    def test_model_generation(self):
        console.print("[bold blue]🤖 Testing Ultron autoregressive text generation...[/bold blue]")
        cfg = UltronConfig(n_layer=2)
        model = UltronModel(cfg)
        unwrapped_model = accelerator.unwrap_model(accelerator.prepare(model))
        unwrapped_model.eval()
        
        prompt = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long, device=device)
        max_new_tokens = 10
        console.print(f"  [cyan]📝 Prompt Shape:[/cyan] [bold white]{prompt.shape}[/bold white] -> Generating [bold yellow]{max_new_tokens}[/bold yellow] new tokens")
        
        generated = unwrapped_model.generate(prompt, max_new_tokens=max_new_tokens)
        console.print(f"  [magenta]✨ Generated Sequence Shape:[/magenta] [bold white]{generated.shape}[/bold white]")
        tt.assert_close(torch.tensor(generated.shape), torch.tensor([1, 8 + max_new_tokens]))
        console.print("[bold green]✅ test_model_generation passed successfully![/bold green]\n")

    def test_rmsnorm(self):
        console.print("[bold blue]🧪 Testing RMSNorm numerical precision and unit variance...[/bold blue]")
        dim = 64
        norm = RMSNorm(dim).to(device)
        x = torch.randn(2, 10, dim, device=device)
        out = norm(x)
        
        console.print(f"  [magenta]📊 RMSNorm Output Shape:[/magenta] [bold white]{out.shape}[/bold white]")
        tt.assert_close(torch.tensor(out.shape), torch.tensor([2, 10, dim]))
        
        mean_sq = out.pow(2).mean(-1)
        console.print(f"  [yellow]⚖️ Head Variance (Target ~ 1.0):[/yellow] min=[bold cyan]{mean_sq.min().item():.4f}[/bold cyan], max=[bold cyan]{mean_sq.max().item():.4f}[/bold cyan]")
        tt.assert_close(mean_sq, torch.ones(2, 10, device=device), rtol=1e-3, atol=1e-3)
        console.print("[bold green]✅ test_rmsnorm passed successfully![/bold green]\n")

    def test_rotary_embedding(self):
        console.print("[bold blue]🔄 Testing Rotary Position Embedding (RoPE) tensor application...[/bold blue]")
        cfg = UltronConfig(n_layer=1)
        model = UltronModel(cfg)
        unwrapped_model = accelerator.unwrap_model(accelerator.prepare(model))
        
        B, T, n_head, head_dim = 2, 16, 12, 64
        q = torch.randn(B, n_head, T, head_dim, device=device)
        console.print(f"  [cyan]📥 Query Shape Before RoPE:[/cyan] [bold white]{q.shape}[/bold white]")
        
        rot_emb = unwrapped_model.rotary_emb(q, T)
        cos, sin = rot_emb
        console.print(f"  [magenta]🌀 RoPE Cos Shape:[/magenta] [bold white]{cos.shape}[/bold white], [magenta]Sin Shape:[/magenta] [bold white]{sin.shape}[/bold white]")
        
        q_rot = apply_rotary_emb(q, cos, sin)
        console.print(f"  [cyan]📤 Query Shape After RoPE:[/cyan] [bold white]{q_rot.shape}[/bold white]")
        tt.assert_close(torch.tensor(q_rot.shape), torch.tensor(q.shape))
        console.print("[bold green]✅ test_rotary_embedding passed successfully![/bold green]\n")

if __name__ == "__main__":
    unittest.main()


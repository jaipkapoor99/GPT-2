"""
Unit tests for Hugging Face wrapper (UltronForCausalLM and UltronHFConfig).
"""

import sys
import os
import tempfile
import unittest
import torch
import torch.testing as tt
from accelerate import Accelerator
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel
from hf_model import UltronHFConfig, UltronForCausalLM

# ── Accelerate guard ────────────────────────────────────────────────────────
if not any(k in os.environ for k in [
    "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
]):
    raise RuntimeError("Run with: accelerate launch -m unittest tests.test_hf_wrapper ...")

console = Console()
accelerator = Accelerator()
device = accelerator.device


class TestUltronHFWrapper(unittest.TestCase):

    def setUp(self):
        console.print(f"[bold cyan]⚙️ Setting up HF wrapper test on device:[/bold cyan] [bold yellow]{device}[/bold yellow]")
        self.hf_config = UltronHFConfig(n_layer=2, n_positions=512)
        self.model_hf = UltronForCausalLM(self.hf_config).to(device)
        self.model_hf.eval()

    def test_weight_tying_data_ptr(self):
        console.print("[bold blue]🔗 Testing Weight Tying (wte.weight and lm_head.weight memory pointer equality)...[/bold blue]")
        wte_ptr = self.model_hf.transformer.wte.weight.data_ptr()
        lm_head_ptr = self.model_hf.lm_head.weight.data_ptr()
        console.print(f"  [cyan]wte.weight data_ptr     :[/cyan] {hex(wte_ptr)}")
        console.print(f"  [cyan]lm_head.weight data_ptr :[/cyan] {hex(lm_head_ptr)}")
        self.assertEqual(wte_ptr, lm_head_ptr)
        console.print("[bold green]✅ test_weight_tying_data_ptr passed successfully![/bold green]\n")

    def test_logits_parity(self):
        console.print("[bold blue]⚡ Testing Logits Parity between native UltronModel and UltronForCausalLM...[/bold blue]")
        native_cfg = self.hf_config.to_ultron_config()
        native_model = UltronModel(native_cfg).to(device)
        native_model.eval()

        # Initialize both with same weights
        self.model_hf.load_state_dict(native_model.state_dict(), strict=False)

        B, T = 2, 32
        input_ids = torch.randint(0, self.hf_config.vocab_size, (B, T), dtype=torch.long, device=device)

        with torch.no_grad():
            native_out = native_model(input_ids)
            hf_out = self.model_hf(input_ids=input_ids)

        console.print(f"  [magenta]Native Logits Shape:[/magenta] {native_out.logits.shape}")
        console.print(f"  [magenta]HF Wrapper Logits Shape:[/magenta] {hf_out.logits.shape}")

        tt.assert_close(native_out.logits, hf_out.logits, rtol=1e-5, atol=1e-5)
        console.print("[bold green]✅ test_logits_parity passed successfully![/bold green]\n")

    def test_save_and_from_pretrained_local_roundtrip(self):
        console.print("[bold blue]💾 Testing save_pretrained and from_pretrained local roundtrip...[/bold blue]")
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model_hf.save_pretrained(tmp_dir)
            console.print(f"  [green]Saved model artifact to:[/green] {tmp_dir}")

            # Copy hf_model.py into tmp_dir so trust_remote_code can load it directly
            hf_model_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_model.py")
            with open(hf_model_src, "r") as f_src, open(os.path.join(tmp_dir, "hf_model.py"), "w") as f_dst:
                f_dst.write(f_src.read())

            loaded_model = UltronForCausalLM.from_pretrained(
                tmp_dir,
                trust_remote_code=True,
                device_map={"": device},
            )
            loaded_model.eval()

            input_ids = torch.randint(0, self.hf_config.vocab_size, (1, 16), dtype=torch.long, device=device)
            with torch.no_grad():
                orig_out = self.model_hf(input_ids=input_ids)
                loaded_out = loaded_model(input_ids=input_ids)

            tt.assert_close(orig_out.logits, loaded_out.logits, rtol=1e-5, atol=1e-5)
        console.print("[bold green]✅ test_save_and_from_pretrained_local_roundtrip passed successfully![/bold green]\n")


if __name__ == "__main__":
    unittest.main()

"""Full-validation metric tests."""

import torch
import torch.nn.functional as F
import pytest

from config import UltronConfig
from model import UltronModel
from scripts.validate import load_checkpoint_weights, sequence_cross_entropy


def tiny_config():
    return UltronConfig(
        B=1,
        T=16,
        C=16,
        n_head=2,
        n_kv_head=1,
        n_layer=1,
        vocab_size=32,
    )


def test_sequence_cross_entropy_matches_per_sequence_reference():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))

    losses = sequence_cross_entropy(logits, targets)
    expected = torch.stack(
        [
            F.cross_entropy(logits[index], targets[index])
            for index in range(len(targets))
        ]
    )

    torch.testing.assert_close(losses, expected)


def test_checkpoint_weight_loader_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model weights found"):
        load_checkpoint_weights(
            UltronModel(tiny_config()),
            tmp_path,
        )


def test_checkpoint_weight_loader_accepts_pytorch_state_dict(tmp_path):
    source = UltronModel(tiny_config())
    weight_path = tmp_path / "pytorch_model.bin"
    torch.save(source.state_dict(), weight_path)
    restored = UltronModel(tiny_config())

    selected = load_checkpoint_weights(restored, tmp_path)

    assert selected == weight_path
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        source.transformer.wte.weight,
    )

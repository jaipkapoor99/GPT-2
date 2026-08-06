"""CPU-safe unit tests for the Ultron model and checkpoint contract."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import (
    RMSNorm,
    UltronModel,
    apply_rotary_emb,
    load_ultron_state_dict,
)


def tiny_config(**overrides):
    values = {
        "B": 2,
        "T": 32,
        "C": 32,
        "n_head": 4,
        "n_kv_head": 2,
        "n_layer": 2,
        "vocab_size": 128,
        "dropout": 0.0,
    }
    values.update(overrides)
    return UltronConfig(**values)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return UltronModel(tiny_config()).eval()


def test_config_defaults():
    config = UltronConfig()
    assert config.C == 768
    assert config.n_head == 12
    assert config.n_kv_head == 4
    assert config.head_dim == 64
    assert config.vocab_size == 49152
    assert config.grad_accum_steps == 4


def test_documented_parameter_count():
    with torch.device("meta"):
        model = UltronModel(UltronConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 113_266_944


def test_forward_shape_and_loss(model):
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    targets = torch.randint(0, model.config.vocab_size, (2, 12))
    output = model(inputs, targets=targets)

    assert output.logits.shape == (2, 12, model.config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_future_tokens_do_not_change_prefix_logits(model):
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    changed = inputs.clone()
    changed[:, 7:] = torch.randint(0, model.config.vocab_size, changed[:, 7:].shape)

    with torch.no_grad():
        original_logits = model(inputs).logits
        changed_logits = model(changed).logits

    torch.testing.assert_close(original_logits[:, :7], changed_logits[:, :7])


def test_cached_decoding_matches_full_forward(model):
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))

    with torch.no_grad():
        full_logits = model(inputs).logits
        cache = None
        cached_logits = []
        for position in range(inputs.size(1)):
            output = model(
                inputs[:, position : position + 1],
                use_cache=True,
                past_key_values=cache,
            )
            cache = output.past_key_values
            cached_logits.append(output.logits)

    torch.testing.assert_close(
        torch.cat(cached_logits, dim=1),
        full_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_checkpoint_loader_allows_only_tied_alias(model):
    state_dict = dict(model.state_dict())
    state_dict.pop("lm_head.weight")

    restored = UltronModel(tiny_config())
    missing, unexpected = load_ultron_state_dict(restored, state_dict)

    assert missing == ["lm_head.weight"]
    assert unexpected == []
    assert restored.transformer.wte.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        model.transformer.wte.weight,
    )


def test_checkpoint_loader_rejects_other_missing_keys(model):
    state_dict = dict(model.state_dict())
    state_dict.pop("transformer.h.0.attn.c_attn.weight")

    with pytest.raises(RuntimeError, match="Incompatible Ultron checkpoint"):
        load_ultron_state_dict(UltronModel(tiny_config()), state_dict)


def test_optimizer_partition_is_complete_and_disjoint(model):
    partitions = model.partition_optimizer_parameters()
    grouped = [parameter for group in partitions.values() for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert len(grouped) == len(trainable)
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }
    assert model.transformer.wte.weight in partitions["adamw_nodecay"]
    assert all(parameter.ndim == 2 for parameter in partitions["muon"])
    assert all(parameter.ndim < 2 for parameter in partitions["adamw_decay"])


def test_repeated_batch_can_be_learned():
    torch.manual_seed(7)
    config = tiny_config(C=16, n_head=2, n_kv_head=1, n_layer=1, vocab_size=32)
    model = UltronModel(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    inputs = torch.arange(12).remainder(config.vocab_size).unsqueeze(0)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    with torch.no_grad():
        initial_loss = F.cross_entropy(
            model(inputs).logits.flatten(0, 1),
            targets.flatten(),
        )

    for _ in range(20):
        optimizer.zero_grad()
        loss = model(inputs, targets=targets).loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = model(inputs, targets=targets).loss

    assert final_loss < initial_loss * 0.5


def test_rmsnorm_has_unit_mean_square():
    norm = RMSNorm(64)
    output = norm(torch.randn(2, 10, 64))
    mean_square = output.pow(2).mean(-1)
    torch.testing.assert_close(
        mean_square,
        torch.ones_like(mean_square),
        rtol=1e-3,
        atol=1e-3,
    )


def test_rotary_embedding_preserves_shape(model):
    query = torch.randn(2, model.config.n_head, 16, model.config.head_dim)
    cosine, sine = model.rotary_emb(query, 16)
    rotated = apply_rotary_emb(query, cosine, sine)
    assert rotated.shape == query.shape


@pytest.mark.skipif(
    os.environ.get("ULTRON_TEST_COMPILE") != "1",
    reason="set ULTRON_TEST_COMPILE=1 to run the slower compiler smoke test",
)
def test_torch_compile_forward(model):
    compiled = torch.compile(model)
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    assert compiled(inputs).logits.shape == (2, 12, model.config.vocab_size)

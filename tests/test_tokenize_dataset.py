"""Exact-resume and atomic-shard helper tests."""

import json

import numpy as np
import pytest

from config import UltronConfig
from scripts.tokenize_dataset import (
    _atomic_json_write,
    _atomic_shard_write,
    _is_tokenization_complete,
    _load_resume_state,
    _new_state,
    _save_resume_state,
)


def make_state(shard_size=8, max_shards=2):
    return _new_state(
        config=UltronConfig(),
        dataset_revision="dataset-sha",
        tokenizer_revision="tokenizer-sha",
        shard_size_tokens=shard_size,
        max_shards=max_shards,
    )


def test_resume_state_round_trip_preserves_exact_pending_tokens(tmp_path):
    state = make_state()
    pending = [3, 5, 8, 13]
    _save_resume_state(tmp_path, state, pending, previous_pending_file=None)

    restored_state, restored_pending = _load_resume_state(
        tmp_path,
        8,
        2,
        UltronConfig(),
    )

    assert restored_state["source_documents_consumed"] == 0
    assert restored_pending.tolist() == pending


def test_committed_shard_size_is_validated(tmp_path):
    state = make_state()
    values = np.arange(8, dtype=np.uint16)
    _atomic_shard_write(tmp_path / "fineweb_edu_shard_0000.bin", values)
    _atomic_json_write(
        tmp_path / "fineweb_edu_shard_0000_meta.json",
        {"shard_index": 0, "tokens": 8, "dtype": "uint16"},
    )
    state.update(
        {
            "next_shard": 1,
            "committed_tokens": 8,
            "source_documents_consumed": 2,
        }
    )
    _save_resume_state(tmp_path, state, [21], previous_pending_file=None)

    restored_state, restored_pending = _load_resume_state(
        tmp_path,
        8,
        2,
        UltronConfig(),
    )

    assert restored_state["next_shard"] == 1
    assert restored_pending.tolist() == [21]

    (tmp_path / "fineweb_edu_shard_0000.bin").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="invalid size"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_outputs_without_resume_state_are_rejected(tmp_path):
    metadata = tmp_path / "fineweb_edu_shard_0000_meta.json"
    metadata.write_text(json.dumps({"shard_index": 0}))

    with pytest.raises(RuntimeError, match="without an exact resume checkpoint"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_complete_state_is_detected_without_reopening_dataset():
    state = make_state()
    state.update({"next_shard": 2, "committed_tokens": 16})

    assert _is_tokenization_complete(state, 8, 2)
    assert not _is_tokenization_complete(state, 8, 3)

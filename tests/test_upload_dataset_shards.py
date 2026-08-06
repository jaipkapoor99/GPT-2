"""Dataset uploader validation tests."""

import json

import numpy as np
import pytest

from scripts.upload_dataset_shards import validate_complete_shard_set


def write_complete_set(directory, shard_count=2, shard_size=8):
    state = {
        "max_shards": shard_count,
        "next_shard": shard_count,
        "shard_size_tokens": shard_size,
        "committed_tokens": shard_count * shard_size,
        "dataset_revision": "dataset-sha",
        "tokenizer_revision": "tokenizer-sha",
    }
    (directory / "tokenization_state.json").write_text(json.dumps(state))
    for index in range(shard_count):
        np.arange(shard_size, dtype=np.uint16).tofile(
            directory / f"fineweb_edu_shard_{index:04d}.bin"
        )
        metadata = {
            "shard_index": index,
            "tokens": shard_size,
            "dtype": "uint16",
            "dataset_revision": "dataset-sha",
            "tokenizer_revision": "tokenizer-sha",
        }
        (directory / f"fineweb_edu_shard_{index:04d}_meta.json").write_text(
            json.dumps(metadata)
        )
    return state


def test_complete_shard_set_is_accepted(tmp_path):
    expected = write_complete_set(tmp_path)

    assert validate_complete_shard_set(tmp_path) == expected


def test_incomplete_shard_set_is_rejected(tmp_path):
    state = write_complete_set(tmp_path)
    state["next_shard"] = 1
    (tmp_path / "tokenization_state.json").write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="incomplete"):
        validate_complete_shard_set(tmp_path)


def test_inconsistent_metadata_is_rejected(tmp_path):
    write_complete_set(tmp_path)
    metadata_path = tmp_path / "fineweb_edu_shard_0001_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_revision"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_complete_shard_set(tmp_path)

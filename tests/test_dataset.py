"""Dataset splitting and shard lookup regression tests."""

import numpy as np
import pytest

from dataset import ZeroCopyShardedDataset, split_train_dev_datasets


def write_shard(path, start, length):
    np.arange(start, start + length, dtype=np.uint16).tofile(path)
    return str(path)


def test_shard_lookup_handles_boundaries_and_negative_indices(tmp_path):
    shards = [
        write_shard(tmp_path / "first.bin", 0, 20),
        write_shard(tmp_path / "second.bin", 100, 20),
    ]
    dataset = ZeroCopyShardedDataset(shards, sequence_length=4, step=4)

    assert dataset[0][0].tolist() == [0, 1, 2, 3]
    assert dataset[4][0].tolist() == [100, 101, 102, 103]
    assert dataset[-1][0].tolist() == [112, 113, 114, 115]
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_multiple_shards_split_at_shard_boundary(tmp_path):
    shards = [
        write_shard(tmp_path / f"{index}.bin", index * 100, 20)
        for index in range(3)
    ]

    train_ds, dev_ds = split_train_dev_datasets(
        shards,
        sequence_length=4,
        step=4,
    )

    assert train_ds.bin_shards == shards[:2]
    assert dev_ds.bin_shards == shards[2:]


def test_single_shard_split_leaves_non_overlapping_gap(tmp_path):
    shard = write_shard(tmp_path / "only.bin", 0, 200)
    train_ds, dev_ds = split_train_dev_datasets(
        [shard],
        sequence_length=8,
        step=2,
    )

    last_train_index = train_ds.indices[-1]
    first_dev_index = dev_ds.indices[0]
    last_train_token = last_train_index * 2 + 8
    first_dev_token = first_dev_index * 2

    assert last_train_token < first_dev_token

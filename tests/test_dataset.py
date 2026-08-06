"""Dataset splitting and shard lookup regression tests."""

import numpy as np
import pytest
from torch.utils.data import SequentialSampler

from config import UltronConfig
from dataset import (
    EpochRandomSampler,
    ZeroCopyShardedDataset,
    get_dataloaders,
    split_train_dev_datasets,
)


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


def test_default_stride_produces_adjacent_non_overlapping_windows(tmp_path):
    shard = write_shard(tmp_path / "tokens.bin", 0, 17)
    dataset = ZeroCopyShardedDataset([shard], sequence_length=4)

    assert dataset.step == 4
    assert len(dataset) == 4
    assert dataset[0][0].tolist() == [0, 1, 2, 3]
    assert dataset[1][0].tolist() == [4, 5, 6, 7]
    assert dataset[-1][1].tolist() == [13, 14, 15, 16]


@pytest.mark.parametrize(
    ("sequence_length", "step"),
    [(0, None), (-1, None), (4, 0), (4, -1)],
)
def test_dataset_rejects_invalid_window_geometry(
    tmp_path,
    sequence_length,
    step,
):
    shard = write_shard(tmp_path / "tokens.bin", 0, 20)

    with pytest.raises(ValueError, match="greater than zero"):
        ZeroCopyShardedDataset(
            [shard],
            sequence_length=sequence_length,
            step=step,
        )


def test_windows_never_cross_shard_boundaries(tmp_path):
    shards = [
        write_shard(tmp_path / "first.bin", 0, 13),
        write_shard(tmp_path / "second.bin", 100, 13),
    ]
    dataset = ZeroCopyShardedDataset(shards, sequence_length=4)

    assert len(dataset) == 6
    assert dataset[2][0].tolist() == [8, 9, 10, 11]
    assert dataset[3][0].tolist() == [100, 101, 102, 103]


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


def test_dataloaders_shuffle_train_and_preserve_dev_order(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards_edu"
    shard_dir.mkdir()
    for index in range(2):
        write_shard(
            shard_dir / f"fineweb_edu_shard_{index:05d}.bin",
            index * 1_000,
            128,
        )

    class FakeAccelerator:
        def print(self, _message):
            pass

        def prepare(self, *loaders):
            return loaders

    monkeypatch.chdir(tmp_path)
    config = UltronConfig(B=2, T=8)
    train_loader, dev_loader = get_dataloaders(config, FakeAccelerator())

    assert isinstance(train_loader.sampler, EpochRandomSampler)
    assert isinstance(dev_loader.sampler, SequentialSampler)
    assert train_loader.drop_last is True
    assert train_loader.dataset.step == config.T
    assert dev_loader.dataset.step == config.T


def test_epoch_random_sampler_is_reproducible_and_changes_each_epoch():
    dataset = list(range(100))
    first = EpochRandomSampler(dataset, seed=42)
    second = EpochRandomSampler(dataset, seed=42)

    epoch_zero = list(first)
    assert epoch_zero == list(second)
    assert sorted(epoch_zero) == dataset

    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero

    with pytest.raises(ValueError, match="negative"):
        first.set_epoch(-1)

"""Complete training-state uploader regression tests."""

import sys

from scripts import upload_checkpoint


def test_checkpoint_uploader_does_not_filter_training_state(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "accelerate_checkpoint"
    checkpoint_dir.mkdir()
    expected_files = {
        "model.safetensors",
        "optimizer.bin",
        "optimizer_1.bin",
        "scheduler.bin",
        "random_states_0.pkl",
        "training_state.json",
    }
    for filename in expected_files:
        (checkpoint_dir / filename).touch()

    class FakeApi:
        def __init__(self):
            self.upload_kwargs = None

        def create_repo(self, **_kwargs):
            pass

        def upload_folder(self, **kwargs):
            self.upload_kwargs = kwargs

    api = FakeApi()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(upload_checkpoint, "login", lambda token: None)
    monkeypatch.setattr(upload_checkpoint, "HfApi", lambda token: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_checkpoint.py",
            "--repo-id=test/ultron",
            f"--checkpoint-dir={checkpoint_dir}",
        ],
    )

    upload_checkpoint.main()

    assert {path.name for path in checkpoint_dir.iterdir()} == expected_files
    assert api.upload_kwargs["folder_path"] == str(checkpoint_dir)
    assert "allow_patterns" not in api.upload_kwargs
    assert "ignore_patterns" not in api.upload_kwargs

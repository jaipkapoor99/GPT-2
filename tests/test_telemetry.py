"""Telemetry calculation and metric-schema tests."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from telemetry import (
    RollingRateMeter,
    TokenizationTelemetry,
    UltronTelemetry,
    ValidationTelemetry,
    format_rate,
    wandb_run_name,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeAccelerator:
    gradient_accumulation_steps = 3
    num_processes = 4
    is_main_process = False

    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))


class FakeMainAccelerator(FakeAccelerator):
    is_main_process = True

    def get_tracker(self, name, unwrap=False):
        assert name == "wandb"
        assert unwrap is True
        return SimpleNamespace(run=SimpleNamespace(id="run-123"))


def test_fresh_wandb_run_name_starts_with_timestamp(monkeypatch):
    monkeypatch.setenv("ULTRON_RUN_NAME", "pretraining")
    now = datetime(2026, 8, 6, 19, 33, 56, tzinfo=timezone.utc)

    assert wandb_run_name("fresh", now) == "20260806-193356-pretraining"
    assert wandb_run_name("continue", now) == "pretraining"


def test_rolling_rate_uses_recent_cumulative_samples():
    clock = FakeClock()
    meter = RollingRateMeter(window_seconds=10, clock=clock)

    assert meter.update(100).units_per_second == 0
    clock.advance(2)
    assert meter.update(140).units_per_second == pytest.approx(20)
    clock.advance(20)
    assert meter.update(240).units_per_second == pytest.approx(5)


def test_training_throughput_counts_all_workers():
    clock = FakeClock()
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator, clock=clock)

    telemetry.update_terminal_progress(4)
    clock.advance(2)
    eta = telemetry.update_terminal_progress(6)

    assert telemetry.global_tokens_per_step == 240
    assert telemetry.last_steps_per_sec == pytest.approx(1)
    assert telemetry.last_throughput == pytest.approx(240)
    assert eta == 14


def test_structured_training_metrics_have_no_legacy_duplicates():
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    telemetry.last_throughput = 123
    telemetry.last_steps_per_sec = 4

    telemetry.log_training_step(3, 2.5, 1e-3)

    step, metrics = accelerator.logged[0]
    assert step == 3
    assert metrics["step"] == 3
    assert metrics["train/loss"] == 2.5
    assert metrics["perf/tokens_per_sec"] == 123
    assert "train_loss" not in metrics
    assert "perf/eta_seconds" not in metrics
    assert "train/progress_percent" not in metrics
    assert "perf/global_tokens_per_step" not in metrics


def test_training_metrics_keep_throughput_and_dev_loss_continuous():
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    telemetry.last_throughput = 240
    telemetry.last_steps_per_sec = 1
    telemetry.set_last_dev_loss(2.75)

    telemetry.log_training_step(step=11, loss=3.5, lr=1e-3)
    telemetry.last_throughput = 300
    telemetry.last_steps_per_sec = 1.25
    telemetry.log_training_step(step=12, loss=3.25, lr=9e-4)

    first = accelerator.logged[0][1]
    second = accelerator.logged[1][1]
    assert first["step"] == 11
    assert second["step"] == 12
    assert first["perf/tokens_per_sec"] == 240
    assert second["perf/tokens_per_sec"] == 300
    assert first["eval/dev_loss"] == 2.75
    assert second["eval/dev_loss"] == 2.75


def test_evaluation_logs_interval_average_and_combined_chart(monkeypatch):
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    chart = object()
    monkeypatch.setattr(telemetry, "_build_loss_comparison_chart", lambda: chart)

    telemetry.log_training_step(step=1, loss=4.0, lr=1e-3)
    telemetry.log_training_step(step=2, loss=2.0, lr=1e-3)
    telemetry.log_evaluation(step=3, train_loss=3.0, dev_loss=2.5, lr=1e-3)

    _, metrics = accelerator.logged[-1]
    assert metrics["step"] == 3
    assert metrics["train/average_loss"] == pytest.approx(3.0)
    assert metrics["eval/dev_loss"] == 2.5
    assert metrics["eval/sampled_dev_loss"] == 2.5
    assert metrics["charts/train_vs_dev_loss"] is chart
    assert telemetry._loss_history_steps == [3]
    assert telemetry._average_train_loss_history == [pytest.approx(3.0)]
    assert telemetry._dev_loss_history == [2.5]

    telemetry.log_training_step(step=4, loss=5.0, lr=1e-3)
    telemetry.log_evaluation(step=5, train_loss=1.0, dev_loss=2.25, lr=1e-3)

    _, metrics = accelerator.logged[-1]
    assert metrics["train/average_loss"] == pytest.approx(3.0)
    assert telemetry._loss_history_steps == [3, 5]


def test_metric_definitions_hide_internal_series(monkeypatch):
    definitions = []
    fake_wandb = SimpleNamespace(
        define_metric=lambda name, **options: definitions.append((name, options))
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    UltronTelemetry._define_wandb_metrics(FakeMainAccelerator())

    by_name = {name: options for name, options in definitions}
    assert by_name["step"]["hidden"] is True
    assert by_name["train/*"]["step_metric"] == "step"
    assert by_name["eval/*"]["step_sync"] is True
    assert by_name["eval/sampled_dev_loss"]["hidden"] is True
    assert by_name["eval/dev_loss"]["goal"] == "minimize"


def test_wandb_run_id_uses_unwrapped_tracker():
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, FakeMainAccelerator())

    assert telemetry.get_wandb_run_id() == "run-123"


def test_tokenization_eta_and_validation():
    clock = FakeClock()
    telemetry = TokenizationTelemetry(
        target_tokens=1_000,
        start_tokens=100,
        clock=clock,
        enabled=False,
    )
    clock.advance(2)

    eta = telemetry.update(added_tokens=200, current_total=300)

    assert telemetry.last_tokens_per_second == pytest.approx(100)
    assert eta == 7
    with pytest.raises(ValueError):
        telemetry.update(added_tokens=-1, current_total=300)
    with pytest.raises(ValueError, match="must equal"):
        telemetry.update(added_tokens=5, current_total=310)


def test_validation_telemetry_reports_local_timing_and_throughput():
    clock = FakeClock()
    accelerator = FakeAccelerator()
    telemetry = ValidationTelemetry(
        total_sequences=10,
        sequence_length=20,
        accelerator=accelerator,
        clock=clock,
    )
    clock.advance(2)

    eta = telemetry.update(processed_sequences=4, mean_loss=2.5)
    telemetry.close()

    assert telemetry.last_tokens_per_second == pytest.approx(40)
    assert telemetry.average_tokens_per_second == pytest.approx(40)
    assert telemetry.elapsed_seconds == 2
    assert eta == 3


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(0, "— tok/s"), (1_500, "1.5k tok/s"), (2_500_000, "2.50M tok/s")],
)
def test_format_rate(rate, expected):
    assert format_rate(rate, "tok") == expected

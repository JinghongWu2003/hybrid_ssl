import math

import pytest

from losses.schedulers import AlphaScheduler, AlphaSchedulerConfig


def collect_values(scheduler: AlphaScheduler, epochs: int) -> list[float]:
    return [scheduler.value(epoch) for epoch in range(epochs)]


def test_alpha_scheduler_decreases_below_default_target() -> None:
    cfg = AlphaSchedulerConfig(
        warmup_epochs=5,
        final_alpha=0.4,
        total_epochs=10,
        start_alpha=1.0,
    )
    scheduler = AlphaScheduler(cfg)

    values = collect_values(scheduler, cfg.total_epochs)

    # Warmup should linearly interpolate between start and final alpha.
    for epoch in range(cfg.warmup_epochs):
        expected = cfg.start_alpha + (cfg.final_alpha - cfg.start_alpha) * (epoch / cfg.warmup_epochs)
        assert values[epoch] == pytest.approx(expected, abs=1e-6)

    # After warmup the value should stay at the final alpha.
    for epoch in range(cfg.warmup_epochs, cfg.total_epochs):
        assert values[epoch] == pytest.approx(cfg.final_alpha, abs=1e-6)

    # One step beyond the configured epochs should also equal the final alpha.
    assert scheduler.value(cfg.total_epochs) == pytest.approx(cfg.final_alpha, abs=1e-6)


def test_alpha_scheduler_increases_above_default_target() -> None:
    cfg = AlphaSchedulerConfig(
        warmup_epochs=0,
        final_alpha=1.2,
        total_epochs=8,
        start_alpha=1.0,
    )
    scheduler = AlphaScheduler(cfg)

    values = collect_values(scheduler, cfg.total_epochs)

    remaining_epochs = max(1, cfg.total_epochs - cfg.warmup_epochs)

    for epoch in range(cfg.total_epochs):
        progress = epoch / remaining_epochs
        expected = cfg.final_alpha + (cfg.start_alpha - cfg.final_alpha) * 0.5 * (1 + math.cos(math.pi * progress))
        assert values[epoch] == pytest.approx(expected, abs=1e-6)

    assert values[0] == pytest.approx(cfg.start_alpha, abs=1e-6)
    assert scheduler.value(cfg.total_epochs) == pytest.approx(cfg.final_alpha, abs=1e-6)

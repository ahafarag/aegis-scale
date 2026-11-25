"""Realistic 7-day cloud workload and correlated Spot recall storm trace generator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from aegis_scale.config import ComputePool
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel


@dataclass(frozen=True)
class TraceInterval:
    """Single 15-minute ground-truth workload and Spot preemption realization."""

    step: int
    hour: float
    true_rps: float
    is_flash_crowd: bool
    is_spot_storm: bool
    recalled_pools: set[str]
    warned_pools: list[tuple[str, str]]


def generate_seven_day_production_trace(
    pools: list[ComputePool],
    base_rps: float = 3600.0,
    steps: int = 672,
    seed: int = 2026,
) -> list[TraceInterval]:
    """Generate 7 days (672 x 15-min steps) of non-stationary production traffic and Spot storms.

    Combines:
      1. Dual-harmonic diurnal + weekday/weekend enterprise traffic modulation
      2. Heavy-tailed Pareto/lognormal micro-bursts
      3. Four unannounced flash-crowd traffic surges (+85% to +130% RPS)
      4. Five correlated AZ + instance-family AWS Spot capacity recall storms
    """
    rng = np.random.default_rng(seed)
    copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)
    baseline_draws = copula.sample_scenarios(pools=pools, num_scenarios=steps, seed=seed + 7)

    # Flash crowd windows (step indices)
    flash_windows = [(110, 118), (260, 268), (430, 440), (590, 598)]
    # Correlated Spot recall storm windows: (start, end, target_az, target_families)
    storm_specs = [
        (80, 92, "us-east-1a", {"c6a", "m6a"}),
        (210, 224, "us-east-1b", {"c7g", "m7g"}),
        (340, 355, "us-east-1a", {"c6a", "c7g", "m7g"}),
        (485, 498, "us-east-1c", {"m6i", "c6i"}),
        (592, 604, "us-east-1a", {"c6a", "m6a", "c7g"}),  # Coincides with flash crowd 4
    ]

    trace: list[TraceInterval] = []
    for t in range(steps):
        hour = t * 0.25
        day_phase = 2.0 * np.pi * (hour % 24.0) / 24.0
        diurnal = 1.0 + 0.55 * np.sin(day_phase - np.pi / 2.0) + 0.15 * np.cos(2.0 * day_phase)
        day_of_week = int(hour // 24) % 7
        weekend_factor = 0.78 if day_of_week in (5, 6) else 1.0

        is_flash = any(s <= t <= e for s, e in flash_windows)
        flash_mult = 1.95 if is_flash else 1.0
        heavy_noise = float(rng.lognormal(mean=0.0, sigma=0.075))

        true_rps = float(base_rps * diurnal * weekend_factor * flash_mult * heavy_noise)

        recalled: set[str] = set()
        warned: list[tuple[str, str]] = []
        is_storm = False

        # Background copula interruptions
        for k, p in enumerate(pools):
            if p.market == "spot" and baseline_draws[t, k] > 0.5:
                recalled.add(p.pool_id)

        # Coordinated regional AZ / family capacity storms (with AWS 2-min / rebalance warning on onset)
        for s_start, s_end, target_az, target_fams in storm_specs:
            if s_start <= t <= s_end:
                is_storm = True
                for p in pools:
                    if p.market == "spot" and (p.az == target_az and p.family in target_fams):
                        recalled.add(p.pool_id)
                        warned.append((p.instance_type, p.az))

        trace.append(
            TraceInterval(
                step=t,
                hour=hour,
                true_rps=true_rps,
                is_flash_crowd=is_flash,
                is_spot_storm=is_storm,
                recalled_pools=recalled,
                warned_pools=sorted(set(warned)),
            )
        )

    return trace

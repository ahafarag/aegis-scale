"""Unit tests for the independent multi-server G/G/c Discrete-Event Simulator (DES)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.discrete_event_validation import (
    DES_STRATEGIES,
    GGcDiscreteEventSimulator,
    run_discrete_event_validation,
    simulate_stationary_ggc,
)


@pytest.mark.parametrize("rho", [0.70, 0.80])
def test_mm1_analytical_convergence(rho: float) -> None:
    """Verify DES converges to exact closed-form M/M/1 mean wait, P99 wait, and P(W_q > 0)."""
    mu = 125.0
    res = simulate_stationary_ggc(
        num_servers=1,
        rho=rho,
        per_vcpu_mu=mu,
        arrival_cv2=1.0,
        service_cv2=1.0,
        num_requests=300_000,
        warmup_requests=30_000,
        seed=2026,
    )

    exact_mean_ms = (rho / (mu * (1.0 - rho))) * 1000.0
    exact_p99_ms = (math.log(rho / 0.01) / (mu * (1.0 - rho))) * 1000.0
    exact_p_wait = rho

    assert abs(float(res["des_p_wait"]) - exact_p_wait) < 0.015
    assert abs(float(res["des_mean_wait_ms"]) - exact_mean_ms) / exact_mean_ms < 0.06
    assert abs(float(res["des_p99_wait_ms"]) - exact_p99_ms) / exact_p99_ms < 0.08


def test_backlog_carryover_and_server_resizing() -> None:
    """Verify backlog carries smoothly across intervals and server resizing maintains heap invariant."""
    sim = GGcDiscreteEventSimulator(num_servers=2)

    # Interval 1: 100 requests arriving rapidly at t=0.0..0.099s, each requiring 0.05s on 2 servers
    # Total service demand = 5.0s across 2 servers -> ~2.5s finish time over a 0.5s interval -> ~2.0s backlog
    arr1 = [i * 0.001 for i in range(100)]
    svc1 = [0.05] * 100
    out1 = sim.run_interval(arr1, svc1, interval_sec=0.5, num_servers=2)
    assert float(out1["carried_backlog_ms"]) > 1500.0
    assert len(sim.server_heap) == 2

    # Interval 2: single request at t=0.01s with 1 server after scale-down must observe carried backlog
    out2 = sim.run_interval([0.01], [0.01], interval_sec=0.5, num_servers=1)
    assert len(sim.server_heap) == 1
    assert float(out2["p99_wait_ms"]) > 1500.0

    # Interval 3: scale up to 4 servers and drain with empty arrivals over 5.0s
    out3 = sim.run_interval([], [], interval_sec=5.0, num_servers=4)
    assert len(sim.server_heap) == 4
    assert float(out3["carried_backlog_ms"]) == 0.0


def test_fast_mode_discrete_event_validation(tmp_path: Path) -> None:
    """Verify fast-mode end-to-end execution produces valid JSON, figure, and high AegisScale continuity."""
    out_json = tmp_path / "des_fast.json"
    out_fig = tmp_path / "des_fast.png"

    res = run_discrete_event_validation(
        fast=True,
        output_json=out_json,
        figure_path=out_fig,
    )

    assert out_json.exists()
    assert out_fig.exists() and out_fig.stat().st_size > 10_000

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["mode"] == "fast"
    assert len(loaded["part1_formula_calibration"]["points"]) == 12
    assert set(loaded["part2_trace_replay"].keys()) == {"AWS-ELB-Web", "Azure-LMM-2024"}

    for ds_name in ("AWS-ELB-Web", "Azure-LMM-2024"):
        strats = loaded["part2_trace_replay"][ds_name]["strategies"]
        assert set(strats.keys()) == set(DES_STRATEGIES)
        assert float(strats["AegisScale_Full"]["des_sla_continuity_pct"]) >= 90.0

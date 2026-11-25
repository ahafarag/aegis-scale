"""Tests for real empirical AWS CloudWatch ELB trace ingestion and evaluation."""

from __future__ import annotations

from benchmarks.real_trace_benchmark import (
    DEFAULT_REAL_TRACE,
    load_real_workload_csv,
    run_empirical_aws_trace_benchmark,
)


def test_real_aws_cloudwatch_elb_trace_loader_and_short_eval() -> None:
    assert DEFAULT_REAL_TRACE.exists()
    ts, rps = load_real_workload_csv(DEFAULT_REAL_TRACE, scale_to_mean_rps=3200.0, max_steps=250)
    assert len(ts) == 250
    assert len(rps) == 250
    assert abs(float(rps.mean()) - 3200.0) < 1e-6

    summary = run_empirical_aws_trace_benchmark(DEFAULT_REAL_TRACE, max_steps=200)
    strats = summary["strategies"]
    assert strats["4_AegisScale_ACI_CVaR_MILP"]["savings_vs_isosla_od_pct"] > 30.0
    assert strats["4_AegisScale_ACI_CVaR_MILP"]["sla_continuity_pct"] >= strats["2_Greedy_Cheapest_Spot_IsoSized"]["sla_continuity_pct"]
    assert strats["4_AegisScale_ACI_CVaR_MILP"]["sla_continuity_pct"] > strats["1a_OnDemand_Reactive_HPA"]["sla_continuity_pct"]

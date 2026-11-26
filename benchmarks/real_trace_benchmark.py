"""Empirical benchmark runner for real AWS CloudWatch ELB and Azure/Spot CSV traces.

Evaluates AegisScale on real-world production AWS Elastic Load Balancer request-count
telemetry (4,032 5-minute intervals = 14 days from the Numenta Anomaly Benchmark
realAWSCloudwatch corpus) against both lagging reactive HPA and iso-SLA 100% On-Demand.

Usage:
    python benchmarks/real_trace_benchmark.py
    python benchmarks/real_trace_benchmark.py --max-steps 300
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.config import SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus

DEFAULT_REAL_TRACE = ROOT / "benchmarks" / "data" / "real_aws_elb_cloudwatch.csv"


def load_real_workload_csv(
    csv_path: Path,
    scale_to_mean_rps: float = 3200.0,
    max_steps: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load a real cloud telemetry CSV (supports 'value', 'rps', or 'request_count' columns)."""
    timestamps: list[str] = []
    raw_values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = row.get("timestamp") or row.get("time") or str(len(timestamps))
            val_str = row.get("rps") or row.get("value") or row.get("request_count")
            if val_str is None:
                raise ValueError(f"Unsupported CSV columns: {list(row.keys())}")
            timestamps.append(ts)
            raw_values.append(max(1.0, float(val_str)))
            if max_steps is not None and len(raw_values) >= max_steps:
                break

    arr = np.asarray(raw_values, dtype=np.float64)
    warmup = max(1, min(60, len(arr)))
    emp_mean = float(np.mean(arr[:warmup]))
    scaled = arr * (scale_to_mean_rps / max(emp_mean, 1e-9))
    return timestamps, scaled


def run_empirical_aws_trace_benchmark(
    csv_path: Path = DEFAULT_REAL_TRACE,
    max_steps: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate 5 provisioning strategies on real 5-minute AWS CloudWatch ELB traffic."""
    out = output_dir or (ROOT / "output")
    out.mkdir(parents=True, exist_ok=True)

    timestamps, rps_trace = load_real_workload_csv(
        csv_path=csv_path,
        scale_to_mean_rps=3200.0,
        max_steps=max_steps,
    )
    n_steps = len(rps_trace)
    dt_hours = 5.0 / 60.0  # 5-minute CloudWatch intervals

    pools = default_us_east_1_pools()
    pool_map = {p.pool_id: p for p in pools}
    policy = SLAPolicy(
        alpha_target=0.015,
        aci_gamma=0.05,
        aci_window=144,
        on_demand_floor_ratio=0.25,
        max_spot_pool_share=0.22,
        max_cvar_shortfall_vcpu=0.0,
        cvar_beta=0.95,
        num_scenarios=200,
    )
    engine = AegisScaleEngine(pools=pools, policy=policy, seed=404)

    copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)
    realized_draws = copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=909)

    cheapest_od = min((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    spot_sorted = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_spot = spot_sorted[0]
    top6_spot = spot_sorted[:6]

    strategies = [
        "1a_OnDemand_Reactive_HPA",
        "1b_OnDemand_IsoSLA_Conformal",
        "2_Greedy_Cheapest_Spot_IsoSized",
        "3_Static_30_70_Mixed_IsoSized",
        "4_AegisScale_ACI_CVaR_MILP",
    ]
    costs = {s: 0.0 for s in strategies}
    breaches = {s: 0 for s in strategies}
    storm_breaches = {s: 0 for s in strategies}
    solve_ms: list[float] = []
    upper_bounds: list[float] = []
    aegis_hourly: list[float] = []
    od_isosla_hourly: list[float] = []

    prev_rps = float(rps_trace[0])

    for t in range(n_steps):
        true_rps = float(rps_trace[t])
        recalled_ids = {
            pools[k].pool_id for k in range(len(pools))
            if pools[k].market == "spot" and realized_draws[t, k] > 0.5
        }

        storm_active = (t % 550) in range(200, 218)
        engine.clear_pool_warnings()
        if storm_active:
            target_az = ["us-east-1a", "us-east-1b", "us-east-1c"][(t // 550) % 3]
            for p in pools:
                if p.market == "spot" and p.az == target_az and p.family in {"c6a", "c7g", "m6a"}:
                    recalled_ids.add(p.pool_id)
                    engine.flag_pool_preemption_warning(p.instance_type, p.az, elevated_prob=0.98)

        report = engine.run_cycle(observed_rps=prev_rps)
        solve_ms.append(report.portfolio.solve_time_ms)
        upper_bounds.append(report.conformal.upper_bound)

        safe_vcpu = report.required_vcpus
        reactive_vcpu = max(
            required_safe_vcpus(
                prev_rps,
                policy.per_vcpu_rps_capacity,
                policy.p99_queue_delay_sla_ms,
                0.99,
                policy.arrival_cv2,
                policy.service_cv2,
            ),
            prev_rps / (0.55 * policy.per_vcpu_rps_capacity),
        )

        # 1a. Standard lagging Reactive On-Demand HPA (55% CPU target)
        alloc_od_react = {cheapest_od.pool_id: max(1, int(np.ceil(reactive_vcpu / cheapest_od.vcpu)))}

        # 1b. Iso-SLA 100% On-Demand (provisioned to exact same conformal safe vCPU requirement)
        alloc_od_isosla = {cheapest_od.pool_id: max(1, int(np.ceil(safe_vcpu / cheapest_od.vcpu)))}

        # 2. Iso-Sized Greedy Cheapest Spot (10% OD, 90% single cheapest Spot pool)
        gr_od = max(1, int(np.ceil(0.10 * safe_vcpu / cheapest_od.vcpu)))
        gr_sp = max(0, int(np.ceil(0.90 * safe_vcpu / cheapest_spot.vcpu)))
        alloc_greedy = {cheapest_od.pool_id: gr_od, cheapest_spot.pool_id: gr_sp}

        # 3. Iso-Sized Static 30/70 Mixed Heuristic across top-6 Spot pools (no copula or warning evacuation)
        st_od = max(1, int(np.ceil(0.30 * safe_vcpu / cheapest_od.vcpu)))
        rem = max(0.0, safe_vcpu - st_od * cheapest_od.vcpu)
        alloc_static = {cheapest_od.pool_id: st_od}
        for sp in top6_spot:
            cnt = int(np.ceil((rem / len(top6_spot)) / sp.vcpu))
            if cnt > 0:
                alloc_static[sp.pool_id] = cnt

        allocs = {
            "1a_OnDemand_Reactive_HPA": alloc_od_react,
            "1b_OnDemand_IsoSLA_Conformal": alloc_od_isosla,
            "2_Greedy_Cheapest_Spot_IsoSized": alloc_greedy,
            "3_Static_30_70_Mixed_IsoSized": alloc_static,
            "4_AegisScale_ACI_CVaR_MILP": report.portfolio.allocations,
        }

        for s_name, alloc in allocs.items():
            h_cost = sum(pool_map[pid].hourly_price_usd * c for pid, c in alloc.items() if c > 0)
            surv_vcpu = sum(
                pool_map[pid].vcpu * c
                for pid, c in alloc.items()
                if c > 0 and (pool_map[pid].market == "on-demand" or pid not in recalled_ids)
            )
            w_ms = (
                ggc_quantile_wait_seconds(
                    true_rps,
                    float(surv_vcpu),
                    policy.per_vcpu_rps_capacity,
                    0.99,
                    policy.arrival_cv2,
                    policy.service_cv2,
                )
                * 1000.0
            )
            costs[s_name] += h_cost * dt_hours
            if w_ms > policy.p99_queue_delay_sla_ms:
                breaches[s_name] += 1
                if storm_active:
                    storm_breaches[s_name] += 1
            if s_name == "4_AegisScale_ACI_CVaR_MILP":
                aegis_hourly.append(h_cost)
            elif s_name == "1b_OnDemand_IsoSLA_Conformal":
                od_isosla_hourly.append(h_cost)

        prev_rps = true_rps

    total_hours = n_steps * dt_hours
    annual_mult = (24.0 * 365.0) / total_hours
    od_isosla_total = costs["1b_OnDemand_IsoSLA_Conformal"]

    summary: dict[str, dict] = {
        "dataset": {
            "source": "Real AWS CloudWatch ELB Request Count (Numenta NAB realAWSCloudwatch)",
            "intervals": n_steps,
            "duration_hours": round(total_hours, 2),
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "peak_to_mean_ratio": round(float(np.max(rps_trace) / np.mean(rps_trace)), 3),
        },
        "strategies": {},
    }
    for s_name in strategies:
        c_val = costs[s_name]
        summary["strategies"][s_name] = {
            "period_cost_usd": round(c_val, 2),
            "annualized_cost_usd": round(c_val * annual_mult, 2),
            "savings_vs_isosla_od_pct": round((od_isosla_total - c_val) / od_isosla_total * 100.0, 2)
            if od_isosla_total > 0
            else 0.0,
            "sla_continuity_pct": round((1.0 - breaches[s_name] / n_steps) * 100.0, 2),
            "total_sla_breaches": breaches[s_name],
            "spot_storm_sla_breaches": storm_breaches[s_name],
        }

    summary["strategies"]["4_AegisScale_ACI_CVaR_MILP"]["median_milp_solve_ms"] = round(float(np.median(solve_ms)), 3)
    summary["strategies"]["4_AegisScale_ACI_CVaR_MILP"]["aci_empirical_miscoverage_pct"] = round(
        engine.predictor.total_errors / max(1, engine.predictor.total_steps) * 100.0, 3
    )

    (out / "real_aws_elb_benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hours = np.arange(n_steps) * dt_hours
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax1.plot(hours, rps_trace, label="Real AWS ELB Traffic (Scaled RPS)", color="#1d4ed8", lw=0.85, alpha=0.8)
    ax1.plot(hours, upper_bounds, label="AegisScale ACI 98.5% Upper Envelope", color="#10b981", lw=0.9, ls="--", alpha=0.85)
    ax1.set_ylabel("Requests / sec")
    ax1.set_title("Empirical Validation on Real 14-Day AWS CloudWatch ELB Traffic (4,032 Intervals)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours, od_isosla_hourly, label="Iso-SLA 100% On-Demand ($/hr)", color="#64748b", lw=0.85, alpha=0.85)
    ax2.plot(hours, aegis_hourly, label="AegisScale CVaR Portfolio ($/hr)", color="#10b981", lw=1.0)
    ax2.set_xlabel("Elapsed Time (Hours — April 10 to April 24 Production Trace)")
    ax2.set_ylabel("Hourly Spend (USD)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "real_aws_elb_trajectory.png", dpi=160)
    docs_fig = ROOT / "docs" / "figures"
    docs_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(docs_fig / "real_aws_elb_trajectory.png", dpi=160)
    plt.close(fig)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AegisScale benchmark on real AWS CloudWatch ELB trace.")
    parser.add_argument("--workload-csv", type=Path, default=DEFAULT_REAL_TRACE)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    res = run_empirical_aws_trace_benchmark(csv_path=args.workload_csv, max_steps=args.max_steps)
    print("\n=== Real AWS CloudWatch ELB Benchmark Summary ===")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

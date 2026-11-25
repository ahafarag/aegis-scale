"""Reproducible 4-way benchmark suite evaluating AegisScale against industry baselines.

Run from repository root:
    .venv\\Scripts\\python.exe benchmarks/run_benchmarks.py
"""

from __future__ import annotations

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
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus
from benchmarks.workload_traces import TraceInterval, generate_seven_day_production_trace


def _evaluate_allocation(
    alloc: dict[str, int],
    pool_map: dict[str, ComputePool],
    interval: TraceInterval,
    policy: SLAPolicy,
) -> tuple[float, int, float, bool]:
    """Return (hourly_cost_usd, surviving_vcpus, realized_p99_wait_ms, sla_breached)."""
    hourly_cost = 0.0
    surv_vcpu = 0
    for pid, count in alloc.items():
        if count <= 0:
            continue
        p = pool_map[pid]
        hourly_cost += p.hourly_price_usd * count
        if p.market == "on-demand" or pid not in interval.recalled_pools:
            surv_vcpu += p.vcpu * count

    wait_ms = (
        ggc_quantile_wait_seconds(
            arrival_rate_rps=interval.true_rps,
            vcpus=float(surv_vcpu),
            per_vcpu_mu=policy.per_vcpu_rps_capacity,
            quantile=0.99,
            arrival_cv2=policy.arrival_cv2,
            service_cv2=policy.service_cv2,
        )
        * 1000.0
    )
    breached = wait_ms > policy.p99_queue_delay_sla_ms
    return hourly_cost, surv_vcpu, wait_ms, breached


def run_benchmark_suite(output_dir: Path | None = None) -> dict:
    """Execute 7-day head-to-head benchmark across 4 provisioning strategies."""
    out = output_dir or (ROOT / "output")
    out.mkdir(parents=True, exist_ok=True)

    pools = default_us_east_1_pools()
    pool_map = {p.pool_id: p for p in pools}
    policy = SLAPolicy(
        alpha_target=0.015,
        aci_gamma=0.06,
        aci_window=96,
        on_demand_floor_ratio=0.25,
        max_spot_pool_share=0.22,
        max_cvar_shortfall_vcpu=0.0,
        cvar_beta=0.95,
        num_scenarios=300,
    )

    trace = generate_seven_day_production_trace(pools=pools, base_rps=3600.0, steps=672, seed=2026)
    engine = AegisScaleEngine(pools=pools, policy=policy, seed=2026)

    cheapest_od = min((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    spot_pools = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_spot = spot_pools[0]
    top6_spot = spot_pools[:6]

    strategies = [
        "1_OnDemand_Reactive_HPA",
        "2_Greedy_Cheapest_Spot",
        "3_Static_30_70_Mixed_Heuristic",
        "4_AegisScale_ACI_CVaR_MILP",
    ]
    totals_cost = {s: 0.0 for s in strategies}
    breaches = {s: 0 for s in strategies}
    storm_breaches = {s: 0 for s in strategies}
    solve_times_ms: list[float] = []

    hourly_series = {s: [] for s in strategies}
    rps_series: list[float] = []
    upper_rps_series: list[float] = []

    dt_hours = 0.25
    prev_rps = trace[0].true_rps

    for interval in trace:
        # Update AegisScale hazard flags from AWS 2-min / rebalance warnings
        engine.clear_pool_warnings()
        for inst_type, az in interval.warned_pools:
            engine.flag_pool_preemption_warning(inst_type, az, elevated_prob=0.98)

        report = engine.run_cycle(observed_rps=prev_rps)
        solve_times_ms.append(report.portfolio.solve_time_ms)

        # Standard AWS Target-Tracking HPA target (55% average utilization target on lagging metric)
        reactive_vcpu = max(
            required_safe_vcpus(
                arrival_rate_rps=prev_rps,
                per_vcpu_mu=policy.per_vcpu_rps_capacity,
                sla_wait_ms=policy.p99_queue_delay_sla_ms,
                quantile=0.99,
                arrival_cv2=policy.arrival_cv2,
                service_cv2=policy.service_cv2,
            ),
            prev_rps / (0.55 * policy.per_vcpu_rps_capacity),
        )

        # Strategy 1: 100% On-Demand Reactive HPA
        od_units = max(1, int(np.ceil(reactive_vcpu / cheapest_od.vcpu)))
        alloc_od = {cheapest_od.pool_id: od_units}

        # Strategy 2: Greedy Cheapest Spot (10% OD base, 90% single cheapest Spot pool)
        gr_od = max(1, int(np.ceil(0.10 * reactive_vcpu / cheapest_od.vcpu)))
        gr_sp = max(0, int(np.ceil(0.90 * reactive_vcpu / cheapest_spot.vcpu)))
        alloc_greedy = {cheapest_od.pool_id: gr_od, cheapest_spot.pool_id: gr_sp}

        # Strategy 3: Static 30/70 Mixed Heuristic across top-6 cheapest Spot pools (ignores copula & warnings)
        st_od = max(1, int(np.ceil(0.30 * reactive_vcpu / cheapest_od.vcpu)))
        rem_vcpu = max(0.0, reactive_vcpu - st_od * cheapest_od.vcpu)
        per_pool_vcpu = rem_vcpu / len(top6_spot)
        alloc_static = {cheapest_od.pool_id: st_od}
        for sp in top6_spot:
            cnt = int(np.ceil(per_pool_vcpu / sp.vcpu))
            if cnt > 0:
                alloc_static[sp.pool_id] = cnt

        # Strategy 4: AegisScale
        alloc_aegis = report.portfolio.allocations

        alloc_by_strat = {
            "1_OnDemand_Reactive_HPA": alloc_od,
            "2_Greedy_Cheapest_Spot": alloc_greedy,
            "3_Static_30_70_Mixed_Heuristic": alloc_static,
            "4_AegisScale_ACI_CVaR_MILP": alloc_aegis,
        }

        for s_name, alloc in alloc_by_strat.items():
            h_cost, _, _, is_breach = _evaluate_allocation(alloc, pool_map, interval, policy)
            totals_cost[s_name] += h_cost * dt_hours
            hourly_series[s_name].append(h_cost)
            if is_breach:
                breaches[s_name] += 1
                if interval.is_spot_storm:
                    storm_breaches[s_name] += 1

        rps_series.append(interval.true_rps)
        upper_rps_series.append(report.conformal.upper_bound)
        prev_rps = interval.true_rps

    n_steps = len(trace)
    annual_mult = 365.0 / 7.0
    od_weekly = totals_cost["1_OnDemand_Reactive_HPA"]

    summary: dict[str, dict] = {}
    for s_name in strategies:
        w_cost = totals_cost[s_name]
        ann_cost = w_cost * annual_mult
        sav_pct = (od_weekly - w_cost) / od_weekly * 100.0 if od_weekly > 0 else 0.0
        cont_pct = (1.0 - breaches[s_name] / n_steps) * 100.0
        summary[s_name] = {
            "weekly_cost_usd": round(w_cost, 2),
            "annualized_cost_usd": round(ann_cost, 2),
            "annualized_savings_vs_od_usd": round((od_weekly - w_cost) * annual_mult, 2),
            "cost_reduction_vs_od_pct": round(sav_pct, 2),
            "sla_continuity_pct": round(cont_pct, 2),
            "total_sla_breaches": breaches[s_name],
            "spot_storm_sla_breaches": storm_breaches[s_name],
        }

    summary["4_AegisScale_ACI_CVaR_MILP"]["median_milp_solve_ms"] = round(float(np.median(solve_times_ms)), 3)
    summary["4_AegisScale_ACI_CVaR_MILP"]["p99_milp_solve_ms"] = round(float(np.quantile(solve_times_ms, 0.99)), 3)

    # Save JSON summary
    json_path = out / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot 1: Executive Pareto Frontier (Annualized Cost vs. SLA Continuity %)
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2))
    labels_pretty = {
        "1_OnDemand_Reactive_HPA": "100% On-Demand HPA",
        "2_Greedy_Cheapest_Spot": "Greedy Cheapest Spot",
        "3_Static_30_70_Mixed_Heuristic": "Static 30/70 Mixed Spot",
        "4_AegisScale_ACI_CVaR_MILP": "AegisScale (ACI + CVaR MILP)",
    }
    colors = {
        "1_OnDemand_Reactive_HPA": "#64748b",
        "2_Greedy_Cheapest_Spot": "#ef4444",
        "3_Static_30_70_Mixed_Heuristic": "#f59e0b",
        "4_AegisScale_ACI_CVaR_MILP": "#10b981",
    }
    for s_name in strategies:
        x_cost = summary[s_name]["annualized_cost_usd"] / 1000.0
        y_cont = summary[s_name]["sla_continuity_pct"]
        ax1.scatter(x_cost, y_cont, s=160, color=colors[s_name], zorder=4, edgecolors="black", linewidths=0.8)
        if s_name == "1_OnDemand_Reactive_HPA":
            offset = (-108, -14)
        elif s_name == "3_Static_30_70_Mixed_Heuristic":
            offset = (9, -12)
        else:
            offset = (9, 4)
        ax1.annotate(
            labels_pretty[s_name],
            (x_cost, y_cont),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold" if "AegisScale" in s_name else "normal",
        )
    ax1.margins(x=0.14, y=0.10)
    ax1.set_xlabel("Annualized Compute Spend ($ Thousands USD — Lower is Better)")
    ax1.set_ylabel("P99 Latency SLA Continuity (% — Higher is Better)")
    ax1.set_title("7-Day Production Benchmark: Cloud Spend vs. SLA Continuity Frontier")
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(out / "pareto_cost_vs_continuity.png", dpi=160)
    plt.close(fig1)

    # Plot 2: 7-Day Workload, ACI Upper Envelope, and Hourly Spend Trajectory
    hours = [iv.hour for iv in trace]
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    ax_top.plot(hours, rps_series, label="True Incoming Workload (RPS)", color="#2563eb", alpha=0.75, lw=1.1)
    ax_top.plot(hours, upper_rps_series, label="AegisScale ACI 98.5% Upper Bound", color="#10b981", ls="--", lw=1.1)
    ax_top.set_ylabel("Requests / sec")
    ax_top.set_title("Adaptive Conformal Traffic Envelope & Hourly Cloud Spend (168 Hours)")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(hours, hourly_series["1_OnDemand_Reactive_HPA"], label="100% On-Demand HPA ($/hr)", color="#64748b", alpha=0.8)
    ax_bot.plot(hours, hourly_series["4_AegisScale_ACI_CVaR_MILP"], label="AegisScale CVaR Portfolio ($/hr)", color="#10b981", lw=1.3)
    ax_bot.set_xlabel("Elapsed Production Time (Hours)")
    ax_bot.set_ylabel("Hourly Spend (USD)")
    ax_bot.legend(loc="upper right")
    ax_bot.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "seven_day_trajectory.png", dpi=160)
    plt.close(fig2)

    return summary


def main() -> int:
    summary = run_benchmark_suite()
    print("\n=== AegisScale 7-Day Production Benchmark Summary ===")
    for name, metrics in summary.items():
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"  {k:<32}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

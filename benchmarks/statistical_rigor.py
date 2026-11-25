"""Multi-dataset empirical benchmark, 30-seed statistical rigor, ablation, and sensitivity suite.

Evaluates AegisScale across four real public Numenta Anomaly Benchmark (NAB) cloud and
viral flash-crowd traces with zero per-dataset hyperparameter tuning, performs 30-seed
Monte Carlo paired Wilcoxon signed-rank hypothesis testing, executes a 5-way architectural
ablation study, and sweeps AZ recall correlation, Kubernetes node cold-start boot latency,
and HiGHS MILP solver scalability up to K = 210 candidate pools.

Usage:
    .venv\\Scripts\\python.exe benchmarks/statistical_rigor.py
    .venv\\Scripts\\python.exe benchmarks/statistical_rigor.py --fast
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus

DATA_DIR = ROOT / "benchmarks" / "data"
DEFAULT_OUTPUT_JSON = ROOT / "output" / "phase2_statistical_results.json"
DEFAULT_FIGURES_DIR = ROOT / "docs" / "figures"

STRATEGY_NAMES = [
    "Reactive_HPA_55",
    "IsoSLA_100_OnDemand",
    "Greedy_Cheapest_Spot",
    "Static_30_70_Mixed",
    "AegisScale_Full",
]

ABLATION_NAMES = [
    "Full_AegisScale",
    "Ablation_No_ACI",
    "Ablation_Static_Conformal",
    "Ablation_No_GGc",
    "Ablation_Uncorrelated_Spot",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and deterministic window alignment for a real empirical NAB trace."""

    name: str
    filename: str
    description: str
    start_offset_full: int
    start_offset_fast: int


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="AWS-ELB-Web",
        filename="real_aws_elb_cloudwatch.csv",
        description="AWS CloudWatch Elastic Load Balancer HTTP RequestCount (NAB realAWSCloudwatch)",
        start_offset_full=0,
        start_offset_fast=0,
    ),
    DatasetSpec(
        name="AWS-EC2-Network",
        filename="real_aws_ec2_network_in.csv",
        description="AWS EC2 NetworkIn ingress burst telemetry (NAB realAWSCloudwatch)",
        start_offset_full=0,
        start_offset_fast=0,
    ),
    DatasetSpec(
        name="AWS-RDS-Compute",
        filename="real_aws_rds_cpu.csv",
        description="AWS RDS database CPU step-shift and sustained load regime (NAB realAWSCloudwatch)",
        start_offset_full=2800,
        start_offset_fast=3040,
    ),
    DatasetSpec(
        name="Viral-Flash-Traffic",
        filename="real_flash_crowd_twitter.csv",
        description="Real viral flash-crowd event stream with >20x Pareto surges (NAB realTweets AAPL)",
        start_offset_full=12600,
        start_offset_fast=13060,
    ),
]


def default_publication_policy(num_scenarios: int = 150) -> SLAPolicy:
    """Return the single fixed zero-retuning SLAPolicy used across all datasets."""
    return SLAPolicy(
        alpha_target=0.015,
        aci_gamma=0.05,
        on_demand_floor_ratio=0.25,
        max_spot_pool_share=0.22,
        max_cvar_shortfall_vcpu=0.0,
        cvar_beta=0.95,
        num_scenarios=num_scenarios,
    )


def load_empirical_trace(
    csv_path: Path,
    scale_to_mean_rps: float = 3200.0,
    max_steps: int = 600,
    start_offset: int = 0,
) -> tuple[list[str], np.ndarray, int]:
    """Load a contiguous slice of a real NAB CSV trace and normalize to target mean RPS."""
    all_timestamps: list[str] = []
    all_values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = row.get("timestamp") or row.get("time") or str(len(all_timestamps))
            val_str = row.get("value") or row.get("rps") or row.get("request_count")
            if val_str is None:
                raise ValueError(f"Unsupported CSV schema in {csv_path}: {list(row.keys())}")
            all_timestamps.append(ts)
            all_values.append(max(1.0, float(val_str)))

    total_rows = len(all_values)
    if total_rows == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")

    effective_start = max(0, min(start_offset, max(0, total_rows - max_steps)))
    sliced_ts = all_timestamps[effective_start : effective_start + max_steps]
    arr = np.asarray(all_values[effective_start : effective_start + max_steps], dtype=np.float64)
    emp_mean = float(np.mean(arr))
    scaled = arr * (scale_to_mean_rps / max(emp_mean, 1e-9))
    return sliced_ts, scaled, total_rows


def _build_storm_schedule(
    rps_trace: np.ndarray,
    seed: int,
) -> list[tuple[int, int, str, str, set[str]]]:
    """Build correlated regional AZ + compute-family Spot recall storms aligned with demand peaks."""
    rng = np.random.default_rng(seed)
    n_steps = len(rps_trace)
    az_choices = ["us-east-1a", "us-east-1b", "us-east-1c"]
    storm_fams = {"c6a", "c7g", "m6a"}

    search_start = min(18, max(2, n_steps // 5))
    candidate_indices = list(np.argsort(-rps_trace[search_start:]) + search_start)
    target_count = 2 if n_steps <= 120 else 3
    min_gap = max(12, n_steps // (target_count + 2))

    chosen_peaks: list[int] = []
    for idx in candidate_indices:
        c_idx = int(idx)
        if all(abs(c_idx - prev) >= min_gap for prev in chosen_peaks):
            chosen_peaks.append(c_idx)
            if len(chosen_peaks) >= target_count:
                break
    chosen_peaks.sort()

    storms: list[tuple[int, int, str, str, set[str]]] = []
    half_w = 4 if n_steps <= 120 else 7
    for s_i, pk in enumerate(chosen_peaks):
        jitter = int(rng.integers(-1, 2))
        s_start = max(2, pk - half_w + jitter)
        s_end = min(n_steps - 1, pk + half_w + jitter)
        az1 = az_choices[(seed + s_i) % len(az_choices)]
        az2 = az_choices[(seed + s_i + 1) % len(az_choices)]
        storms.append((s_start, s_end, az1, az2, storm_fams))
    return storms


def _apply_boot_delay(
    target_alloc: dict[str, int],
    ready_alloc: dict[str, int],
    pending_queues: dict[str, list[list[int]]],
    step: int,
    boot_delay_steps: int,
    all_pool_ids: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Advance Kubernetes node boot queue by one step and return (ready_alloc, billed_alloc).

    Newly requested instances enter pending_queues with ready_step = step + boot_delay_steps
    and cannot serve requests until ready_step, while scale-downs cancel newest pending
    launches first before terminating active nodes.
    """
    if boot_delay_steps <= 0 or step == 0:
        ready_now = {pid: int(target_alloc.get(pid, 0)) for pid in all_pool_ids if target_alloc.get(pid, 0) > 0}
        ready_alloc.clear()
        ready_alloc.update(ready_now)
        for pid in all_pool_ids:
            pending_queues[pid] = []
        return dict(ready_alloc), dict(ready_alloc)

    billed_now: dict[str, int] = {}
    for pid in all_pool_ids:
        q = pending_queues.setdefault(pid, [])
        still_pending: list[list[int]] = []
        promoted = 0
        for entry in q:
            r_step, cnt = entry[0], entry[1]
            if r_step <= step:
                promoted += cnt
            else:
                still_pending.append(entry)
        cur_ready = ready_alloc.get(pid, 0) + promoted
        cur_pending = sum(item[1] for item in still_pending)
        scheduled = cur_ready + cur_pending
        desired = max(0, int(target_alloc.get(pid, 0)))

        if desired > scheduled:
            still_pending.append([step + boot_delay_steps, desired - scheduled])
        elif desired < scheduled:
            excess = scheduled - desired
            while excess > 0 and still_pending:
                newest = still_pending[-1]
                take = min(excess, newest[1])
                newest[1] -= take
                excess -= take
                if newest[1] <= 0:
                    still_pending.pop()
            if excess > 0:
                cur_ready = max(0, cur_ready - excess)

        pending_queues[pid] = still_pending
        if cur_ready > 0:
            ready_alloc[pid] = cur_ready
        else:
            ready_alloc.pop(pid, None)

        tot_billed = cur_ready + sum(item[1] for item in still_pending)
        if tot_billed > 0:
            billed_now[pid] = tot_billed

    return dict(ready_alloc), billed_now


def _evaluate_five_strategies_trajectory(
    rps_trace: np.ndarray,
    policy: SLAPolicy,
    seed: int,
    env_az_corr: float = 0.55,
    env_fam_corr: float = 0.40,
    boot_delay_steps: int = 0,
    record_coldstart_only: bool = False,
) -> dict[str, Any]:
    """Simulate all 5 provisioning strategies on a given RPS trace and Spot recall realization."""
    pools = default_us_east_1_pools()
    pool_map = {p.pool_id: p for p in pools}
    all_pids = [p.pool_id for p in pools]
    n_steps = len(rps_trace)
    dt_hours = 5.0 / 60.0

    engine = AegisScaleEngine(pools=pools, policy=policy, seed=seed)
    copula = CorrelatedSpotHazardModel(az_correlation=env_az_corr, family_correlation=env_fam_corr)
    realized_draws = copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=seed + 503)
    storms = _build_storm_schedule(rps_trace=rps_trace, seed=seed + 19)

    od_sorted = sorted((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_od = od_sorted[0]
    spot_sorted = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_spot = spot_sorted[0]
    top6_spot = spot_sorted[:6]

    active_strats = ["Reactive_HPA_55", "AegisScale_Full"] if record_coldstart_only else STRATEGY_NAMES
    costs = {s: 0.0 for s in active_strats}
    breaches = {s: 0 for s in active_strats}
    storm_breaches = {s: 0 for s in active_strats}
    p99_waits: dict[str, list[float]] = {s: [] for s in active_strats}
    solve_ms: list[float] = []

    ready_state: dict[str, dict[str, int]] = {s: {} for s in active_strats}
    pending_state: dict[str, dict[str, list[list[int]]]] = {s: {} for s in active_strats}

    prev_rps = float(rps_trace[0])

    for t in range(n_steps):
        true_rps = float(rps_trace[t])
        recalled_ids = {
            pools[k].pool_id
            for k in range(len(pools))
            if pools[k].market == "spot" and realized_draws[t, k] > 0.5
        }

        storm_active = False
        engine.clear_pool_warnings()
        for s_start, s_end, target_az1, target_az2, target_fams in storms:
            if s_start <= t <= s_end:
                storm_active = True
                for p in pools:
                    if p.market == "spot" and (
                        (p.az == target_az1 and p.family in target_fams)
                        or (p.az == target_az2 and p.family in {"c6a", "c7g"})
                    ):
                        recalled_ids.add(p.pool_id)
                        engine.flag_pool_preemption_warning(p.instance_type, p.az, elevated_prob=0.98)

        report = engine.run_cycle(observed_rps=prev_rps)
        solve_ms.append(report.portfolio.solve_time_ms)
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

        alloc_reactive = {cheapest_od.pool_id: max(1, int(math.ceil(reactive_vcpu / cheapest_od.vcpu)))}
        alloc_aegis = report.portfolio.allocations

        if record_coldstart_only:
            target_by_strat = {
                "Reactive_HPA_55": alloc_reactive,
                "AegisScale_Full": alloc_aegis,
            }
        else:
            alloc_isosla_od = {cheapest_od.pool_id: max(1, int(math.ceil(safe_vcpu / cheapest_od.vcpu)))}
            gr_od = max(1, int(math.ceil(0.10 * safe_vcpu / cheapest_od.vcpu)))
            gr_sp = max(0, int(math.ceil(0.90 * safe_vcpu / cheapest_spot.vcpu)))
            alloc_greedy = {cheapest_od.pool_id: gr_od, cheapest_spot.pool_id: gr_sp}

            st_od = max(1, int(math.ceil(0.30 * safe_vcpu / cheapest_od.vcpu)))
            rem = max(0.0, safe_vcpu - st_od * cheapest_od.vcpu)
            total_spot_inst = int(math.ceil(rem / 4.0))
            base_cnt, extra_cnt = divmod(total_spot_inst, len(top6_spot))
            alloc_static = {cheapest_od.pool_id: st_od}
            for idx_sp, sp in enumerate(top6_spot):
                cnt = base_cnt + (1 if idx_sp < extra_cnt else 0)
                if cnt > 0:
                    alloc_static[sp.pool_id] = cnt

            target_by_strat = {
                "Reactive_HPA_55": alloc_reactive,
                "IsoSLA_100_OnDemand": alloc_isosla_od,
                "Greedy_Cheapest_Spot": alloc_greedy,
                "Static_30_70_Mixed": alloc_static,
                "AegisScale_Full": alloc_aegis,
            }

        for s_name, target_alloc in target_by_strat.items():
            ready_alloc, billed_alloc = _apply_boot_delay(
                target_alloc=target_alloc,
                ready_alloc=ready_state[s_name],
                pending_queues=pending_state[s_name],
                step=t,
                boot_delay_steps=boot_delay_steps,
                all_pool_ids=all_pids,
            )
            h_cost = sum(pool_map[pid].hourly_price_usd * c for pid, c in billed_alloc.items() if c > 0)
            surv_vcpu = sum(
                pool_map[pid].vcpu * c
                for pid, c in ready_alloc.items()
                if c > 0 and (pool_map[pid].market == "on-demand" or pid not in recalled_ids)
            )
            w_sec = ggc_quantile_wait_seconds(
                arrival_rate_rps=true_rps,
                vcpus=float(surv_vcpu),
                per_vcpu_mu=policy.per_vcpu_rps_capacity,
                quantile=0.99,
                arrival_cv2=policy.arrival_cv2,
                service_cv2=policy.service_cv2,
            )
            w_ms = min(5000.0, w_sec * 1000.0) if math.isfinite(w_sec) else 5000.0
            p99_waits[s_name].append(w_ms)
            costs[s_name] += h_cost * dt_hours
            if w_sec * 1000.0 > policy.p99_queue_delay_sla_ms:
                breaches[s_name] += 1
                if storm_active:
                    storm_breaches[s_name] += 1

        prev_rps = true_rps

    total_hours = n_steps * dt_hours
    annual_mult = (24.0 * 365.0) / total_hours
    isosla_cost = costs.get("IsoSLA_100_OnDemand", 0.0)

    out_strats: dict[str, dict[str, float | int]] = {}
    for s_name in active_strats:
        c_val = costs[s_name]
        sav_pct = ((isosla_cost - c_val) / isosla_cost * 100.0) if isosla_cost > 0 else 0.0
        out_strats[s_name] = {
            "period_cost_usd": round(c_val, 2),
            "annualized_cost_usd": round(c_val * annual_mult, 2),
            "savings_vs_isosla_od_pct": round(sav_pct, 2),
            "sla_continuity_pct": round((1.0 - breaches[s_name] / n_steps) * 100.0, 2),
            "total_sla_breaches": int(breaches[s_name]),
            "spot_storm_sla_breaches": int(storm_breaches[s_name]),
            "mean_realized_p99_wait_ms": round(float(np.mean(p99_waits[s_name])), 3),
        }

    out_strats["AegisScale_Full"]["median_milp_solve_ms"] = round(float(np.median(solve_ms)), 3)
    out_strats["AegisScale_Full"]["p99_milp_solve_ms"] = round(float(np.quantile(solve_ms, 0.99)), 3)
    out_strats["AegisScale_Full"]["aci_empirical_miscoverage_pct"] = round(
        engine.predictor.total_errors / max(1, engine.predictor.total_steps) * 100.0, 3
    )
    return out_strats


def _run_ablation_variant(
    variant_name: str,
    rps_trace: np.ndarray,
    base_policy: SLAPolicy,
    seed: int = 404,
) -> dict[str, float | int]:
    """Run one of the 5 architectural ablation variants on an empirical RPS trace."""
    pools = default_us_east_1_pools()
    pool_map = {p.pool_id: p for p in pools}
    n_steps = len(rps_trace)
    dt_hours = 5.0 / 60.0

    if variant_name == "Ablation_Static_Conformal":
        pol = base_policy.model_copy(update={"aci_gamma": 1e-9})
    elif variant_name == "Ablation_Uncorrelated_Spot":
        pol = base_policy.model_copy(
            update={
                "az_correlation": 0.0,
                "family_correlation": 0.0,
                "max_spot_pool_share": 1.0,
                "on_demand_floor_ratio": 0.10,
            }
        )
    else:
        pol = base_policy

    predictor = AdaptiveConformalPredictor(
        alpha_target=pol.alpha_target,
        gamma=pol.aci_gamma,
        window_size=pol.aci_window,
    )
    hazard_model = CorrelatedSpotHazardModel(
        az_correlation=pol.az_correlation,
        family_correlation=pol.family_correlation,
    )
    optimizer = CVaRPortfolioOptimizer(policy=pol)

    env_copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)
    realized_draws = env_copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=seed + 503)
    storms = _build_storm_schedule(rps_trace=rps_trace, seed=seed + 19)

    current_alloc: dict[str, int] | None = None
    elevated_hazards: dict[str, float] = {}
    total_cost = 0.0
    breaches = 0
    storm_breaches = 0
    prev_rps = float(rps_trace[0])

    for t in range(n_steps):
        true_rps = float(rps_trace[t])
        recalled_ids = {
            pools[k].pool_id
            for k in range(len(pools))
            if pools[k].market == "spot" and realized_draws[t, k] > 0.5
        }
        storm_active = False
        elevated_hazards.clear()
        for s_start, s_end, target_az1, target_az2, target_fams in storms:
            if s_start <= t <= s_end:
                storm_active = True
                for p in pools:
                    if p.market == "spot" and (
                        (p.az == target_az1 and p.family in target_fams)
                        or (p.az == target_az2 and p.family in {"c6a", "c7g"})
                    ):
                        recalled_ids.add(p.pool_id)
                        if variant_name != "Ablation_Uncorrelated_Spot":
                            elevated_hazards[p.pool_id] = 0.98

        conformal_res = predictor.step(prev_rps)
        next_point, next_upper = predictor.predict_upper_bound()

        if variant_name == "Ablation_No_ACI":
            design_rps = max(1.0, conformal_res.point_forecast * 1.05, next_point * 1.05)
        elif variant_name == "Ablation_Static_Conformal":
            # Static split/sliding conformal quantile without online alpha_t update or max-envelope guard
            design_rps = max(1.0, next_upper)
        else:
            design_rps = max(conformal_res.upper_bound, next_upper, prev_rps)

        if variant_name == "Ablation_No_GGc":
            c_req = max(1.0, design_rps / (0.85 * pol.per_vcpu_rps_capacity))
        else:
            c_req = required_safe_vcpus(
                arrival_rate_rps=design_rps,
                per_vcpu_mu=pol.per_vcpu_rps_capacity,
                sla_wait_ms=pol.p99_queue_delay_sla_ms,
                quantile=0.99,
                arrival_cv2=pol.arrival_cv2,
                service_cv2=pol.service_cv2,
            )

        scenarios = hazard_model.sample_scenarios(
            pools=pools,
            num_scenarios=pol.num_scenarios,
            elevated_pool_hazards=elevated_hazards,
            seed=seed + t,
        )
        decision = optimizer.solve(
            pools=pools,
            required_vcpus=c_req,
            scenarios=scenarios,
            previous_allocations=current_alloc,
        )
        current_alloc = dict(decision.allocations)

        h_cost = sum(pool_map[pid].hourly_price_usd * c for pid, c in current_alloc.items() if c > 0)
        surv_vcpu = sum(
            pool_map[pid].vcpu * c
            for pid, c in current_alloc.items()
            if c > 0 and (pool_map[pid].market == "on-demand" or pid not in recalled_ids)
        )
        w_sec = ggc_quantile_wait_seconds(
            arrival_rate_rps=true_rps,
            vcpus=float(surv_vcpu),
            per_vcpu_mu=pol.per_vcpu_rps_capacity,
            quantile=0.99,
            arrival_cv2=pol.arrival_cv2,
            service_cv2=pol.service_cv2,
        )
        total_cost += h_cost * dt_hours
        if w_sec * 1000.0 > pol.p99_queue_delay_sla_ms:
            breaches += 1
            if storm_active:
                storm_breaches += 1

        prev_rps = true_rps

    total_hours = n_steps * dt_hours
    annual_mult = (24.0 * 365.0) / total_hours
    return {
        "period_cost_usd": round(total_cost, 2),
        "annualized_cost_usd": round(total_cost * annual_mult, 2),
        "sla_continuity_pct": round((1.0 - breaches / n_steps) * 100.0, 2),
        "total_sla_breaches": int(breaches),
        "spot_storm_sla_breaches": int(storm_breaches),
    }


def _worker_dataset_task(args: tuple[DatasetSpec, int, int]) -> tuple[str, dict[str, Any]]:
    """Worker entrypoint for Experiment A single-dataset evaluation."""
    spec, max_steps, num_scenarios = args
    offset = spec.start_offset_fast if max_steps <= 120 else spec.start_offset_full
    timestamps, rps_trace, total_rows = load_empirical_trace(
        csv_path=DATA_DIR / spec.filename,
        scale_to_mean_rps=3200.0,
        max_steps=max_steps,
        start_offset=offset,
    )
    policy = default_publication_policy(num_scenarios=num_scenarios)
    strat_metrics = _evaluate_five_strategies_trajectory(
        rps_trace=rps_trace,
        policy=policy,
        seed=404,
    )
    return spec.name, {
        "metadata": {
            "csv_file": spec.filename,
            "description": spec.description,
            "total_csv_rows": total_rows,
            "evaluated_steps": len(rps_trace),
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "mean_rps": round(float(np.mean(rps_trace)), 2),
            "peak_rps": round(float(np.max(rps_trace)), 2),
            "peak_to_mean_ratio": round(float(np.max(rps_trace) / np.mean(rps_trace)), 3),
        },
        "strategies": strat_metrics,
    }


def _worker_mc_seed_task(args: tuple[np.ndarray, int, int]) -> dict[str, dict[str, float | int]]:
    """Worker entrypoint for Experiment B single Monte Carlo seed trajectory."""
    base_elb_trace, seed_idx, num_scenarios = args
    seed = 1000 + seed_idx * 97
    rng = np.random.default_rng(seed)
    n_steps = len(base_elb_trace)

    # Independent heavy-tailed micro-burst modulation around the real AWS ELB trace
    burst_mult = rng.lognormal(mean=0.0, sigma=0.085, size=n_steps)
    spike_pos = int(rng.integers(max(10, n_steps // 5), max(15, n_steps - 10)))
    burst_mult[spike_pos : min(n_steps, spike_pos + 3)] *= float(rng.uniform(1.55, 1.95))
    perturbed = base_elb_trace * burst_mult
    perturbed = perturbed * (3200.0 / float(np.mean(perturbed)))

    policy = default_publication_policy(num_scenarios=num_scenarios)
    return _evaluate_five_strategies_trajectory(
        rps_trace=perturbed,
        policy=policy,
        seed=seed,
    )


def _worker_ablation_task(args: tuple[str, str, np.ndarray, int]) -> tuple[str, str, dict[str, float | int]]:
    """Worker entrypoint for Experiment C single (dataset, ablation_variant) evaluation."""
    ds_name, variant_name, rps_trace, num_scenarios = args
    policy = default_publication_policy(num_scenarios=num_scenarios)
    metrics = _run_ablation_variant(
        variant_name=variant_name,
        rps_trace=rps_trace,
        base_policy=policy,
        seed=404,
    )
    return ds_name, variant_name, metrics


def _worker_az_corr_task(args: tuple[float, np.ndarray, int]) -> tuple[float, dict[str, Any]]:
    """Worker entrypoint for Experiment D AZ correlation sensitivity point."""
    rho_az, elb_trace, num_scenarios = args
    rho_fam = 0.20
    policy = default_publication_policy(num_scenarios=num_scenarios).model_copy(
        update={"az_correlation": rho_az, "family_correlation": rho_fam}
    )
    res = _evaluate_five_strategies_trajectory(
        rps_trace=elb_trace,
        policy=policy,
        seed=404,
        env_az_corr=rho_az,
        env_fam_corr=rho_fam,
    )
    return rho_az, {
        "rho_az": rho_az,
        "rho_fam": rho_fam,
        "AegisScale_sla_continuity_pct": res["AegisScale_Full"]["sla_continuity_pct"],
        "Greedy_Spot_sla_continuity_pct": res["Greedy_Cheapest_Spot"]["sla_continuity_pct"],
        "Static_30_70_sla_continuity_pct": res["Static_30_70_Mixed"]["sla_continuity_pct"],
        "AegisScale_annualized_cost_usd": res["AegisScale_Full"]["annualized_cost_usd"],
        "AegisScale_spot_storm_breaches": res["AegisScale_Full"]["spot_storm_sla_breaches"],
        "Greedy_Spot_spot_storm_breaches": res["Greedy_Cheapest_Spot"]["spot_storm_sla_breaches"],
        "Static_30_70_spot_storm_breaches": res["Static_30_70_Mixed"]["spot_storm_sla_breaches"],
    }


def _worker_coldstart_task(args: tuple[int, np.ndarray, int]) -> tuple[int, dict[str, Any]]:
    """Worker entrypoint for Experiment D Kubernetes node cold-start delay point."""
    boot_delay, elb_trace, num_scenarios = args
    policy = default_publication_policy(num_scenarios=num_scenarios)
    res = _evaluate_five_strategies_trajectory(
        rps_trace=elb_trace,
        policy=policy,
        seed=404,
        boot_delay_steps=boot_delay,
        record_coldstart_only=True,
    )
    return boot_delay, {
        "boot_delay_steps": boot_delay,
        "boot_delay_seconds": boot_delay * 60,
        "AegisScale_Full": res["AegisScale_Full"],
        "Reactive_HPA_55": res["Reactive_HPA_55"],
    }


def build_scaled_pool_catalog(target_k: int) -> list[ComputePool]:
    """Expand the 21-pool reference catalog into target_k multi-AZ/generation pools."""
    base_pools = default_us_east_1_pools()
    base_len = len(base_pools)
    expanded: list[ComputePool] = []
    idx = 0
    while len(expanded) < target_k:
        template = base_pools[idx % base_len]
        replica = idx // base_len
        az_num = (idx % 6) + 1
        az_name = f"us-east-1{chr(ord('a') + (az_num - 1))}"
        price_factor = 1.0 + 0.0029 * (((idx * 11) % 37) - 18) + 0.00017 * idx
        if template.market == "on-demand":
            prob = 0.0
        else:
            prob = min(
                0.25,
                max(0.008, template.base_interruption_prob * (1.0 + 0.013 * (((idx * 7) % 23) - 11))),
            )
        expanded.append(
            ComputePool(
                pool_id=f"{template.pool_id}-r{replica}-k{idx}",
                instance_type=template.instance_type,
                family=f"{template.family}-g{replica % 4}",
                az=az_name,
                market=template.market,
                vcpu=template.vcpu,
                memory_gib=template.memory_gib,
                hourly_price_usd=round(template.hourly_price_usd * price_factor, 6),
                base_interruption_prob=round(prob, 5),
                max_instances=template.max_instances,
            )
        )
        idx += 1
    return expanded


def run_highs_scalability_sweep(
    pool_counts: list[int],
    num_scenarios: int = 200,
    trials_per_k: int = 25,
) -> list[dict[str, float | int]]:
    """Benchmark HiGHS CVaR MILP solve latency across fleet pool counts K = 21..210."""
    policy = default_publication_policy(num_scenarios=num_scenarios)
    optimizer = CVaRPortfolioOptimizer(policy=policy)
    hazard = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)

    results: list[dict[str, float | int]] = []
    for k_count in pool_counts:
        pools = build_scaled_pool_catalog(k_count)
        prev_alloc: dict[str, int] | None = None
        latencies_ms: list[float] = []
        for trial in range(trials_per_k):
            req_vcpus = 96.0 + 32.0 * math.sin(trial * 0.7) + 12.0 * (k_count / 21.0)
            scenarios = hazard.sample_scenarios(
                pools=pools,
                num_scenarios=num_scenarios,
                seed=7000 + k_count * 31 + trial,
            )
            decision = optimizer.solve(
                pools=pools,
                required_vcpus=req_vcpus,
                scenarios=scenarios,
                previous_allocations=prev_alloc,
            )
            prev_alloc = decision.allocations
            latencies_ms.append(decision.solve_time_ms)

        arr = np.asarray(latencies_ms, dtype=np.float64)
        results.append(
            {
                "pool_count_k": int(k_count),
                "num_scenarios": int(num_scenarios),
                "trials": int(trials_per_k),
                "median_ms": round(float(np.median(arr)), 3),
                "p99_ms": round(float(np.quantile(arr, 0.99)), 3),
                "mean_ms": round(float(np.mean(arr)), 3),
                "max_ms": round(float(np.max(arr)), 3),
                "within_1000ms_realtime_budget": bool(float(np.quantile(arr, 0.99)) < 1000.0),
            }
        )
    return results


def _safe_wilcoxon_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Compute exact two-sided paired Wilcoxon signed-rank p-value."""
    diffs = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if np.all(np.abs(diffs) < 1e-12):
        return 1.0
    res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
    return float(res.pvalue)


def _summarize_mc_results(
    per_seed_records: list[dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Compute mean, sample std, 95% CI half-width, and paired Wilcoxon p-values."""
    n_seeds = len(per_seed_records)
    metrics_of_interest = [
        "annualized_cost_usd",
        "savings_vs_isosla_od_pct",
        "sla_continuity_pct",
        "total_sla_breaches",
        "spot_storm_sla_breaches",
    ]
    strategy_stats: dict[str, dict[str, dict[str, float]]] = {}
    raw_arrays: dict[str, dict[str, np.ndarray]] = {s: {} for s in STRATEGY_NAMES}

    for s_name in STRATEGY_NAMES:
        strategy_stats[s_name] = {}
        for m_key in metrics_of_interest:
            vals = np.asarray([float(rec[s_name][m_key]) for rec in per_seed_records], dtype=np.float64)
            raw_arrays[s_name][m_key] = vals
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
            ci95_hw = 1.96 * std_val / math.sqrt(float(n_seeds))
            strategy_stats[s_name][m_key] = {
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "ci95_halfwidth": round(ci95_hw, 3),
            }

    p_cost = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["annualized_cost_usd"],
        raw_arrays["IsoSLA_100_OnDemand"]["annualized_cost_usd"],
    )
    p_cont_greedy = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["sla_continuity_pct"],
        raw_arrays["Greedy_Cheapest_Spot"]["sla_continuity_pct"],
    )
    p_cont_reactive = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["sla_continuity_pct"],
        raw_arrays["Reactive_HPA_55"]["sla_continuity_pct"],
    )
    p_storm_static = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["spot_storm_sla_breaches"],
        raw_arrays["Static_30_70_Mixed"]["spot_storm_sla_breaches"],
    )

    return {
        "n_seeds": n_seeds,
        "dataset": "AWS-ELB-Web",
        "strategy_statistics": strategy_stats,
        "wilcoxon_two_sided_tests": {
            "p_value_cost_vs_isosla_od": p_cost,
            "p_value_continuity_vs_greedy": p_cont_greedy,
            "p_value_continuity_vs_reactive": p_cont_reactive,
            "p_value_storm_vs_static": p_storm_static,
            "all_significant_at_alpha_0_05": bool(
                max(p_cost, p_cont_greedy, p_cont_reactive, p_storm_static) < 0.05
            ),
        },
    }


def generate_publication_figures(
    results: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """Generate the 3 crisp, zero-overlap publication figures at 160 DPI."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
        }
    )

    saved_paths: list[str] = []

    # -------------------------------------------------------------------------
    # Figure 1: multi_dataset_generalization.png (2-panel bar chart)
    # -------------------------------------------------------------------------
    exp_a = results["experiment_a_multi_dataset"]
    ds_names = [spec.name for spec in DATASET_SPECS]
    strat_labels = {
        "Reactive_HPA_55": "Reactive HPA (55%)",
        "IsoSLA_100_OnDemand": "Iso-SLA 100% OD",
        "Greedy_Cheapest_Spot": "Greedy Cheapest Spot",
        "Static_30_70_Mixed": "Static 30/70 Mixed",
        "AegisScale_Full": "AegisScale (Full)",
    }
    strat_colors = {
        "Reactive_HPA_55": "#64748b",
        "IsoSLA_100_OnDemand": "#3b82f6",
        "Greedy_Cheapest_Spot": "#ef4444",
        "Static_30_70_Mixed": "#f59e0b",
        "AegisScale_Full": "#10b981",
    }

    x = np.arange(len(ds_names), dtype=np.float64)
    width = 0.155
    offsets = (np.arange(len(STRATEGY_NAMES)) - 2.0) * width

    fig1, (ax_cont, ax_cost) = plt.subplots(1, 2, figsize=(14.4, 6.1))
    fig1.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.98, wspace=0.18)

    for idx, s_name in enumerate(STRATEGY_NAMES):
        cont_vals = [float(exp_a[ds]["strategies"][s_name]["sla_continuity_pct"]) for ds in ds_names]
        cost_vals = [float(exp_a[ds]["strategies"][s_name]["annualized_cost_usd"]) / 1000.0 for ds in ds_names]
        edge_w = 1.0 if s_name == "AegisScale_Full" else 0.4
        ax_cont.bar(
            x + offsets[idx],
            cont_vals,
            width=width,
            label=strat_labels[s_name],
            color=strat_colors[s_name],
            edgecolor="#0f172a",
            linewidth=edge_w,
        )
        ax_cost.bar(
            x + offsets[idx],
            cost_vals,
            width=width,
            color=strat_colors[s_name],
            edgecolor="#0f172a",
            linewidth=edge_w,
        )

    min_cont = min(
        float(exp_a[ds]["strategies"][s]["sla_continuity_pct"])
        for ds in ds_names
        for s in STRATEGY_NAMES
    )
    ax_cont.set_ylim(max(0.0, math.floor(min_cont - 6.0)), 102.5)
    ax_cont.axhline(98.5, color="#0f172a", linestyle="--", linewidth=1.1, alpha=0.75, label="98.5% Target Envelope")
    ax_cont.set_xticks(x)
    ax_cont.set_xticklabels(ds_names, rotation=10, ha="right")
    ax_cont.set_ylabel("P99 Latency SLA Continuity (%)")
    ax_cont.set_title("(a) Zero-Retuning SLA Continuity Across Real Traces")
    ax_cont.grid(axis="y", linestyle="--", alpha=0.35)

    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(ds_names, rotation=10, ha="right")
    ax_cost.set_ylabel("Annualized Compute Spend ($K USD)")
    ax_cost.set_title("(b) Annualized Provisioning Cost Across Real Traces")
    ax_cost.grid(axis="y", linestyle="--", alpha=0.35)
    ax_cost.margins(y=0.10)

    handles, labels = ax_cont.get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=6,
        framealpha=0.95,
        fontsize=8.8,
    )
    fig1.suptitle(
        "Multi-Dataset Empirical Generalization (Fixed SLAPolicy Across 4 Numenta NAB Production Workloads)",
        y=0.975,
        fontsize=12.5,
        fontweight="bold",
    )
    p1 = figures_dir / "multi_dataset_generalization.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)
    saved_paths.append(str(p1))

    # -------------------------------------------------------------------------
    # Figure 2: ablation_and_coldstart.png (Left = 5-way Ablation; Right = Cold-Start)
    # -------------------------------------------------------------------------
    exp_c = results["experiment_c_ablations"]
    exp_d_cold = results["experiment_d_sensitivity_and_scalability"]["cold_start_delay_sweep"]

    fig2, (ax_abl, ax_cold) = plt.subplots(1, 2, figsize=(14.0, 5.5), constrained_layout=True)

    abl_pretty = {
        "Full_AegisScale": "Full AegisScale",
        "Ablation_No_ACI": "No ACI (Point x1.05)",
        "Ablation_Static_Conformal": "Static Conformal",
        "Ablation_No_GGc": "No G/G/c (Mean Rate)",
        "Ablation_Uncorrelated_Spot": "Uncorrelated Spot",
    }
    abl_colors = {
        "Full_AegisScale": "#10b981",
        "Ablation_No_ACI": "#ef4444",
        "Ablation_Static_Conformal": "#f59e0b",
        "Ablation_No_GGc": "#8b5cf6",
        "Ablation_Uncorrelated_Spot": "#ec4899",
    }
    abl_markers = {
        "Full_AegisScale": "o",
        "Ablation_No_ACI": "s",
        "Ablation_Static_Conformal": "^",
        "Ablation_No_GGc": "D",
        "Ablation_Uncorrelated_Spot": "X",
    }

    # Plot mean across the two ablation datasets (AWS-ELB-Web & Viral-Flash-Traffic) for crisp frontier
    pts_x: list[float] = []
    pts_y: list[float] = []
    for v_name in ABLATION_NAMES:
        c_elb = float(exp_c["AWS-ELB-Web"][v_name]["annualized_cost_usd"]) / 1000.0
        c_vir = float(exp_c["Viral-Flash-Traffic"][v_name]["annualized_cost_usd"]) / 1000.0
        b_elb = float(exp_c["AWS-ELB-Web"][v_name]["total_sla_breaches"])
        b_vir = float(exp_c["Viral-Flash-Traffic"][v_name]["total_sla_breaches"])
        avg_cost_k = 0.5 * (c_elb + c_vir)
        tot_breaches = b_elb + b_vir
        pts_x.append(avg_cost_k)
        pts_y.append(tot_breaches)
        ax_abl.scatter(
            avg_cost_k,
            tot_breaches,
            s=145,
            color=abl_colors[v_name],
            marker=abl_markers[v_name],
            edgecolors="#0f172a",
            linewidths=0.9,
            zorder=4,
            label=f"{abl_pretty[v_name]} ({int(tot_breaches)} breaches, ${avg_cost_k:.1f}K)",
        )

    ax_abl.margins(x=0.18)
    ax_abl.set_ylim(-max(2.0, 0.06 * max(pts_y)), max(12.0, max(pts_y) * 1.48))
    ax_abl.set_xlabel("Mean Annualized Spend Across ELB + Viral Traces ($K USD)")
    ax_abl.set_ylabel("Combined P99 SLA Breaches (Lower is Better)")
    ax_abl.set_title("(a) 5-Way Architectural Ablation: Cost vs. SLA Breaches")
    ax_abl.grid(True, linestyle="--", alpha=0.35)
    ax_abl.legend(loc="upper right", framealpha=0.94)

    delays_sec = [int(item["boot_delay_seconds"]) for item in exp_d_cold]
    aegis_cold = [float(item["AegisScale_Full"]["sla_continuity_pct"]) for item in exp_d_cold]
    react_cold = [float(item["Reactive_HPA_55"]["sla_continuity_pct"]) for item in exp_d_cold]

    ax_cold.plot(
        delays_sec,
        aegis_cold,
        marker="o",
        linewidth=2.3,
        markersize=7.5,
        color="#10b981",
        label="AegisScale (Proactive ACI + Churn Guard)",
    )
    ax_cold.plot(
        delays_sec,
        react_cold,
        marker="s",
        linewidth=2.3,
        markersize=7.5,
        linestyle="--",
        color="#ef4444",
        label="Reactive HPA (55% Target Utilization)",
    )
    for d_s, y_a, y_r in zip(delays_sec, aegis_cold, react_cold):
        ax_cold.annotate(
            f"{y_a:.1f}%",
            (d_s, y_a),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color="#065f46",
            fontweight="bold",
        )
        ax_cold.annotate(
            f"{y_r:.1f}%",
            (d_s, y_r),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color="#991b1b",
        )

    ax_cold.set_xticks(delays_sec)
    ax_cold.set_xticklabels([f"{d}s\n({d // 60} step)" if d == 60 else f"{d}s\n({d // 60} steps)" for d in delays_sec])
    low_y = min(min(react_cold), min(aegis_cold))
    ax_cold.set_ylim(max(0.0, low_y - 9.0), 103.5)
    ax_cold.set_xlim(-15, max(delays_sec) + 18)
    ax_cold.set_xlabel("Kubernetes Node Cold-Start Provisioning Lag")
    ax_cold.set_ylabel("P99 Latency SLA Continuity (%)")
    ax_cold.set_title("(b) Node Cold-Start Provisioning Delay Sensitivity")
    ax_cold.grid(True, linestyle="--", alpha=0.35)
    ax_cold.legend(loc="center right", framealpha=0.93)

    fig2.suptitle(
        "Component Ablation Analysis and Kubernetes Node Cold-Start Resilience",
        fontsize=12.5,
        fontweight="bold",
    )
    p2 = figures_dir / "ablation_and_coldstart.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    saved_paths.append(str(p2))

    # -------------------------------------------------------------------------
    # Figure 3: solver_scalability.png (HiGHS MILP Median & P99 vs K = 21..210)
    # -------------------------------------------------------------------------
    exp_scale = results["experiment_d_sensitivity_and_scalability"]["highs_scalability_sweep"]
    k_vals = [int(item["pool_count_k"]) for item in exp_scale]
    med_ms = [float(item["median_ms"]) for item in exp_scale]
    p99_ms = [float(item["p99_ms"]) for item in exp_scale]

    fig3, ax_sc = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax_sc.plot(
        k_vals,
        med_ms,
        marker="o",
        linewidth=2.2,
        markersize=7.5,
        color="#2563eb",
        label="Median Solve Latency (ms)",
    )
    ax_sc.plot(
        k_vals,
        p99_ms,
        marker="^",
        linewidth=2.2,
        markersize=7.5,
        color="#d97706",
        linestyle="-.",
        label="P99 Tail Solve Latency (ms)",
    )
    ax_sc.axhline(
        1000.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label="Real-Time Control Budget (1,000 ms)",
    )

    for k_v, m_v, p_v in zip(k_vals, med_ms, p99_ms):
        ax_sc.annotate(
            f"{p_v:.1f} ms",
            (k_v, p_v),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#92400e",
        )
        ax_sc.annotate(
            f"{m_v:.1f} ms",
            (k_v, m_v),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#1e40af",
        )

    ax_sc.set_yscale("log")
    ax_sc.set_xticks(k_vals)
    ax_sc.set_xticklabels([str(k) for k in k_vals])
    ax_sc.set_xlim(min(k_vals) - 10, max(k_vals) + 14)
    ax_sc.set_ylim(max(1.0, min(med_ms) * 0.40), max(2200.0, max(p99_ms) * 1.75))
    ax_sc.set_xlabel("Candidate Compute Pool Count K (Multi-AZ / Multi-Family Fleet)")
    ax_sc.set_ylabel("HiGHS CVaR MILP Solve Time (ms, log scale)")
    ax_sc.set_title("Rockafellar-Uryasev CVaR MILP Scalability (S = 200 Scenarios per Solve)")
    ax_sc.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_sc.legend(loc="lower right", framealpha=0.94)

    p3 = figures_dir / "solver_scalability.png"
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)
    saved_paths.append(str(p3))

    return saved_paths


def run_statistical_rigor_suite(
    fast: bool = False,
    max_steps: int | None = None,
    n_seeds: int | None = None,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Execute Experiments A, B, C, and D, write JSON report, and render 3 figures."""
    t_start = time.perf_counter()
    steps = max_steps if max_steps is not None else (80 if fast else 600)
    seeds = n_seeds if n_seeds is not None else (6 if fast else 30)
    num_scenarios = 90 if fast else 150
    scalability_trials = 8 if fast else 25

    _, elb_trace, _ = load_empirical_trace(
        csv_path=DATA_DIR / "real_aws_elb_cloudwatch.csv",
        scale_to_mean_rps=3200.0,
        max_steps=steps,
        start_offset=0,
    )
    viral_offset = 13060 if steps <= 120 else 12600
    _, viral_trace, _ = load_empirical_trace(
        csv_path=DATA_DIR / "real_flash_crowd_twitter.csv",
        scale_to_mean_rps=3200.0,
        max_steps=steps,
        start_offset=viral_offset,
    )

    max_workers = min(os.cpu_count() or 4, 8)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit Experiment A (4 real datasets)
        fut_exp_a = [
            executor.submit(_worker_dataset_task, (spec, steps, num_scenarios))
            for spec in DATASET_SPECS
        ]

        # Submit Experiment B (N independent Monte Carlo seeds on AWS-ELB-Web)
        fut_exp_b = [
            executor.submit(_worker_mc_seed_task, (elb_trace, s_idx, num_scenarios))
            for s_idx in range(seeds)
        ]

        # Submit Experiment C (5-way ablation on AWS-ELB-Web & Viral-Flash-Traffic)
        ablation_tasks: list[tuple[str, str, np.ndarray, int]] = []
        for ds_label, ds_trace in [("AWS-ELB-Web", elb_trace), ("Viral-Flash-Traffic", viral_trace)]:
            for v_name in ABLATION_NAMES:
                ablation_tasks.append((ds_label, v_name, ds_trace, num_scenarios))
        fut_exp_c = [executor.submit(_worker_ablation_task, task) for task in ablation_tasks]

        # Submit Experiment D.1 (AZ correlation sweep) and D.2 (Node cold-start delay sweep)
        rho_az_grid = [0.0, 0.25, 0.55, 0.75]
        fut_az = [
            executor.submit(_worker_az_corr_task, (rho_az, elb_trace, num_scenarios))
            for rho_az in rho_az_grid
        ]
        boot_delays = [0, 1, 2, 3]
        fut_cold = [
            executor.submit(_worker_coldstart_task, (d, elb_trace, num_scenarios))
            for d in boot_delays
        ]

        exp_a_results: dict[str, Any] = {}
        for fut in fut_exp_a:
            ds_name, ds_payload = fut.result()
            exp_a_results[ds_name] = ds_payload

        mc_records = [fut.result() for fut in fut_exp_b]
        exp_b_results = _summarize_mc_results(mc_records)

        exp_c_results: dict[str, dict[str, Any]] = {"AWS-ELB-Web": {}, "Viral-Flash-Traffic": {}}
        for fut in fut_exp_c:
            ds_label, v_name, v_metrics = fut.result()
            exp_c_results[ds_label][v_name] = v_metrics

        az_sweep_map = dict(fut.result() for fut in fut_az)
        az_sweep_list = [az_sweep_map[r] for r in rho_az_grid]

        cold_sweep_map = dict(fut.result() for fut in fut_cold)
        cold_sweep_list = [cold_sweep_map[d] for d in boot_delays]

    # Experiment D.3: Single-thread timing of HiGHS MILP scalability without CPU contention
    scalability_list = run_highs_scalability_sweep(
        pool_counts=[21, 42, 84, 147, 210],
        num_scenarios=200,
        trials_per_k=scalability_trials,
    )

    elapsed_sec = time.perf_counter() - t_start
    fixed_pol = default_publication_policy(num_scenarios=num_scenarios)

    results: dict[str, Any] = {
        "suite_metadata": {
            "mode": "fast" if fast else "publication_full",
            "max_steps_per_dataset": steps,
            "monte_carlo_seeds": seeds,
            "milp_scenarios": num_scenarios,
            "elapsed_wall_time_sec": round(elapsed_sec, 2),
            "fixed_sla_policy": fixed_pol.model_dump(),
        },
        "experiment_a_multi_dataset": exp_a_results,
        "experiment_b_monte_carlo_wilcoxon": exp_b_results,
        "experiment_c_ablations": exp_c_results,
        "experiment_d_sensitivity_and_scalability": {
            "az_correlation_sweep": az_sweep_list,
            "cold_start_delay_sweep": cold_sweep_list,
            "highs_scalability_sweep": scalability_list,
        },
    }

    figure_paths = generate_publication_figures(results=results, figures_dir=figures_dir)
    results["suite_metadata"]["generated_figures"] = figure_paths

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AegisScale multi-dataset, 30-seed Wilcoxon, ablation, and sensitivity benchmark."
    )
    parser.add_argument("--fast", action="store_true", help="Run fast CI/unit-test configuration (80 steps, 6 seeds).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override steps per dataset.")
    parser.add_argument("--n-seeds", type=int, default=None, help="Override number of Monte Carlo seeds.")
    args = parser.parse_args()

    res = run_statistical_rigor_suite(
        fast=args.fast,
        max_steps=args.max_steps,
        n_seeds=args.n_seeds,
    )
    meta = res["suite_metadata"]
    wilc = res["experiment_b_monte_carlo_wilcoxon"]["wilcoxon_two_sided_tests"]
    print(f"AegisScale Statistical Rigor Suite completed in {meta['elapsed_wall_time_sec']}s ({meta['mode']}).")
    print("Wilcoxon two-sided p-values:", json.dumps(wilc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

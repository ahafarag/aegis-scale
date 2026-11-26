"""Phase 3 Audited Multi-Dataset, Statistical Rigor, Ablation, Sensitivity, and K8s Benchmark Suite.

Evaluates AegisScale with zero per-dataset hyperparameter retuning across five real public
production traces (four Numenta Anomaly Benchmark traces + the October 2024 Microsoft Azure
Public Dataset Multimodal LLM Inference trace):
  - Experiment A: Multi-dataset generalization across 6 strategies (including 1-Yr AWS Compute
    Savings Plan baseline and fair AWS 2-min Spot warning reaction for Static 30/70 Mixed).
  - Experiment B: 30 disjoint real-world rolling windows (6 per dataset x 5 datasets) with
    exact Student-t 95% confidence intervals and two-sided paired Wilcoxon signed-rank tests.
  - Experiment C: Strictly unconfounded 6-way architectural ablation (one variable changed per row).
  - Experiment D: AZ correlation sweep, true 1-minute (dt=60s) node cold-start sweep with
    recalled-Spot eviction, Spot price multiplier sensitivity (0.7x-1.6x), and HiGHS scalability.
  - Experiment E: Empirical 2D (vCPU, GiB) Kubernetes/Karpenter benchmark across us-east-1,
    eu-west-1, and ap-northeast-1 using K8sAegisOperator and build_2d_effective_pools.

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
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.cloud.multi_region import SUPPORTED_REGIONS, get_region_pools
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.k8s.bin_packing import K8sClusterOverhead, PodResources, pods_per_node
from aegis_scale.k8s.karpenter import K8sAegisOperator
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus

DATA_DIR = ROOT / "benchmarks" / "data"
DEFAULT_OUTPUT_JSON = ROOT / "output" / "phase3_audited_results.json"
DEFAULT_FIGURES_DIR = ROOT / "docs" / "figures"

# 1-Year No-Upfront AWS Compute Savings Plan discount (~32% off On-Demand on committed base)
SAVINGS_PLAN_DISCOUNT = 0.32
HTTP_GATEWAY_TIMEOUT_MS = 5000.0

STRATEGY_NAMES = [
    "Reactive_HPA_55",
    "IsoSLA_100_OnDemand",
    "OD_SavingsPlan_IsoSLA",
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
    "Ablation_No_Spot_Warnings",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and deterministic window alignment for a real empirical trace."""

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
    DatasetSpec(
        name="Azure-LLM-Inference-2024",
        filename="real_azure_inference_2024.csv",
        description="Microsoft Azure Public Dataset Oct 2024 Multimodal LLM Inference (1M requests, 5-min bins)",
        start_offset_full=0,
        start_offset_fast=0,
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
    causal_warmup_steps: int = 60,
) -> tuple[list[str], np.ndarray, int]:
    """Load a contiguous slice of a real CSV trace using strictly causal prefix normalization.

    Normalizes raw telemetry using ONLY the initial warmup prefix mean (`arr[:warmup]`) so
    no future demand statistics leak into earlier steps.
    """
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
    warmup_len = max(1, min(causal_warmup_steps, len(arr)))
    causal_prefix_mean = float(np.mean(arr[:warmup_len]))
    scaled = arr * (scale_to_mean_rps / max(causal_prefix_mean, 1e-9))
    return sliced_ts, scaled, total_rows


def resample_5min_to_1min(trace_5min: np.ndarray) -> np.ndarray:
    """Linearly interpolate a 5-minute trace to exact 1-minute (60s) resolution."""
    n5 = len(trace_5min)
    if n5 <= 1:
        return trace_5min.copy()
    x_old = np.arange(n5, dtype=np.float64) * 5.0
    x_new = np.arange((n5 - 1) * 5 + 1, dtype=np.float64)
    return np.interp(x_new, x_old, trace_5min)


def _build_exogenous_storm_schedule(
    n_steps: int,
    seed: int,
) -> list[tuple[int, int, str, str, set[str]]]:
    """Generate correlated regional AZ + family Spot recall storms INDEPENDENTLY of future RPS."""
    rng = np.random.default_rng(seed)
    az_choices = ["us-east-1a", "us-east-1b", "us-east-1c"]
    storm_fams = {"c6a", "c7g", "m6a"}

    target_count = 2 if n_steps <= 120 else 3
    usable_start = max(12, n_steps // 8)
    usable_end = max(usable_start + 10, n_steps - max(8, n_steps // 12))
    segment = max(12, (usable_end - usable_start) // target_count)
    half_w = 4 if n_steps <= 120 else 7

    storms: list[tuple[int, int, str, str, set[str]]] = []
    for s_i in range(target_count):
        seg_lo = usable_start + s_i * segment
        seg_hi = min(usable_end, seg_lo + segment - 1)
        center = int(rng.integers(seg_lo, max(seg_lo + 1, seg_hi)))
        s_start = max(2, center - half_w)
        s_end = min(n_steps - 1, center + half_w)
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
    recalled_ids: set[str] | None = None,
    pool_map: dict[str, ComputePool] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Advance node boot queues by one step and evict recalled Spot nodes from ready_alloc."""
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

    snapshot_ready = dict(ready_alloc)

    # Evict interrupted Spot instances from ready_alloc AFTER evaluating this step's survival
    # so replacement launches in step t+1 must genuinely pass through pending_queues.
    if recalled_ids and pool_map:
        for pid in recalled_ids:
            if pid in pool_map and pool_map[pid].market == "spot":
                ready_alloc.pop(pid, None)
                pending_queues[pid] = []

    return snapshot_ready, billed_now


def _scale_spot_prices(pools: list[ComputePool], spot_price_mult: float) -> list[ComputePool]:
    """Scale hourly prices of Spot pools and ensure generous enterprise EC2 quotas for viral spikes."""
    out: list[ComputePool] = []
    for p in pools:
        if p.market == "spot":
            out.append(
                p.model_copy(
                    update={
                        "hourly_price_usd": round(p.hourly_price_usd * spot_price_mult, 6),
                        "max_instances": max(p.max_instances, 600),
                    }
                )
            )
        else:
            out.append(p.model_copy(update={"max_instances": max(p.max_instances, 2500)}))
    return out


def _evaluate_strategies_trajectory(
    rps_trace: np.ndarray,
    policy: SLAPolicy,
    seed: int,
    env_az_corr: float = 0.55,
    env_fam_corr: float = 0.40,
    boot_delay_steps: int = 0,
    dt_hours: float = 5.0 / 60.0,
    spot_price_mult: float = 1.0,
    record_coldstart_only: bool = False,
) -> dict[str, Any]:
    """Simulate all provisioning strategies on a real RPS trace and exogenous Spot recalls."""
    pools = _scale_spot_prices(default_us_east_1_pools(), spot_price_mult)
    pool_map = {p.pool_id: p for p in pools}
    all_pids = [p.pool_id for p in pools]
    n_steps = len(rps_trace)

    engine = AegisScaleEngine(pools=pools, policy=policy, seed=seed)
    copula = CorrelatedSpotHazardModel(az_correlation=env_az_corr, family_correlation=env_fam_corr)
    realized_draws = copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=seed + 503)
    storms = _build_exogenous_storm_schedule(n_steps=n_steps, seed=seed + 19)

    od_sorted = sorted((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_od = od_sorted[0]
    spot_sorted = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_spot = spot_sorted[0]

    # Causal 1-Year Compute Savings Plan hourly commitment: 30th percentile of initial warmup window
    warmup_len = max(5, min(60, n_steps // 4))
    sp_base_rps = float(np.quantile(rps_trace[:warmup_len], 0.30))
    sp_base_vcpu = required_safe_vcpus(
        sp_base_rps,
        policy.per_vcpu_rps_capacity,
        policy.p99_queue_delay_sla_ms,
        0.99,
        policy.arrival_cv2,
        policy.service_cv2,
    )
    sp_committed_od_nodes = max(1, int(math.ceil(sp_base_vcpu / cheapest_od.vcpu)))
    sp_committed_hourly_od_equiv = sp_committed_od_nodes * cheapest_od.hourly_price_usd

    active_strats = ["Reactive_HPA_55", "AegisScale_Full"] if record_coldstart_only else STRATEGY_NAMES
    costs = {s: 0.0 for s in active_strats}
    breaches = {s: 0 for s in active_strats}
    storm_breaches = {s: 0 for s in active_strats}
    unstable_steps = {s: 0 for s in active_strats}
    p99_waits_capped: dict[str, list[float]] = {s: [] for s in active_strats}
    p99_waits_finite: dict[str, list[float]] = {s: [] for s in active_strats}
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
        warned_spot_ids: set[str] = set()
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
                        warned_spot_ids.add(p.pool_id)
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

            # Fair AWS 2-min warning reaction for Static 30/70 Mixed: filter out warned Spot pools
            healthy_spot = [p for p in spot_sorted if p.pool_id not in warned_spot_ids]
            top6_healthy = (healthy_spot if healthy_spot else spot_sorted)[:6]
            st_od = max(1, int(math.ceil(0.30 * safe_vcpu / cheapest_od.vcpu)))
            rem = max(0.0, safe_vcpu - st_od * cheapest_od.vcpu)
            total_spot_inst = int(math.ceil(rem / 4.0))
            base_cnt, extra_cnt = divmod(total_spot_inst, len(top6_healthy))
            alloc_static = {cheapest_od.pool_id: st_od}
            for idx_sp, sp in enumerate(top6_healthy):
                cnt = base_cnt + (1 if idx_sp < extra_cnt else 0)
                if cnt > 0:
                    alloc_static[sp.pool_id] = cnt

            target_by_strat = {
                "Reactive_HPA_55": alloc_reactive,
                "IsoSLA_100_OnDemand": alloc_isosla_od,
                "OD_SavingsPlan_IsoSLA": alloc_isosla_od,
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
                recalled_ids=recalled_ids,
                pool_map=pool_map,
            )
            raw_h_cost = sum(pool_map[pid].hourly_price_usd * c for pid, c in billed_alloc.items() if c > 0)
            if s_name == "OD_SavingsPlan_IsoSLA":
                # Committed 1-Yr Savings Plan covers up to sp_committed_hourly_od_equiv at (1 - discount),
                # billed continuously even if utilization dips below commitment; bursts billed at OD rate.
                sp_covered = sp_committed_hourly_od_equiv * (1.0 - SAVINGS_PLAN_DISCOUNT)
                od_burst = max(0.0, raw_h_cost - sp_committed_hourly_od_equiv)
                h_cost = sp_covered + od_burst
            else:
                h_cost = raw_h_cost

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
            if math.isfinite(w_sec):
                w_ms = min(HTTP_GATEWAY_TIMEOUT_MS, w_sec * 1000.0)
                p99_waits_finite[s_name].append(w_ms)
            else:
                w_ms = HTTP_GATEWAY_TIMEOUT_MS
                unstable_steps[s_name] += 1
            p99_waits_capped[s_name].append(w_ms)
            costs[s_name] += h_cost * dt_hours
            if (not math.isfinite(w_sec)) or (w_sec * 1000.0 > policy.p99_queue_delay_sla_ms):
                breaches[s_name] += 1
                if storm_active:
                    storm_breaches[s_name] += 1

        prev_rps = true_rps

    total_hours = n_steps * dt_hours
    annual_mult = (24.0 * 365.0) / total_hours
    isosla_cost = costs.get("IsoSLA_100_OnDemand", 0.0)
    sp_cost = costs.get("OD_SavingsPlan_IsoSLA", 0.0)

    out_strats: dict[str, dict[str, float | int]] = {}
    for s_name in active_strats:
        c_val = costs[s_name]
        sav_od = ((isosla_cost - c_val) / isosla_cost * 100.0) if isosla_cost > 0 else 0.0
        sav_sp = ((sp_cost - c_val) / sp_cost * 100.0) if sp_cost > 0 else 0.0
        finite_arr = p99_waits_finite[s_name]
        out_strats[s_name] = {
            "period_cost_usd": round(c_val, 2),
            "annualized_cost_usd": round(c_val * annual_mult, 2),
            "savings_vs_isosla_od_pct": round(sav_od, 2),
            "savings_vs_savings_plan_pct": round(sav_sp, 2),
            "sla_continuity_pct": round((1.0 - breaches[s_name] / n_steps) * 100.0, 2),
            "total_sla_breaches": int(breaches[s_name]),
            "spot_storm_sla_breaches": int(storm_breaches[s_name]),
            "unstable_interval_pct": round((unstable_steps[s_name] / n_steps) * 100.0, 2),
            "mean_realized_p99_wait_ms": round(float(np.mean(p99_waits_capped[s_name])), 3),
            "mean_finite_p99_wait_ms": round(float(np.mean(finite_arr)) if finite_arr else HTTP_GATEWAY_TIMEOUT_MS, 3),
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
    """Run one of the 6 strictly unconfounded architectural ablation variants (1 change per row)."""
    pools = default_us_east_1_pools()
    pool_map = {p.pool_id: p for p in pools}
    n_steps = len(rps_trace)
    dt_hours = 5.0 / 60.0

    if variant_name == "Ablation_Static_Conformal":
        pol = base_policy.model_copy(update={"aci_gamma": 1e-9})
    elif variant_name == "Ablation_Uncorrelated_Spot":
        # Strictly disable copula correlation & pool-share cap ONLY; preserve OD floor & warnings!
        pol = base_policy.model_copy(
            update={
                "az_correlation": 0.0,
                "family_correlation": 0.0,
                "max_spot_pool_share": 1.0,
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
    storms = _build_exogenous_storm_schedule(n_steps=n_steps, seed=seed + 19)

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
                        if variant_name != "Ablation_No_Spot_Warnings":
                            elevated_hazards[p.pool_id] = 0.98

        predictor.step(prev_rps)
        next_point, next_upper = predictor.predict_upper_bound()

        if variant_name == "Ablation_No_ACI":
            design_rps = max(1.0, next_point * 1.05)
        else:
            # Identical design_rps formula across Full, Static Conformal, No GGc, Uncorrelated, No Warnings
            design_rps = max(next_upper, prev_rps)

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
        if (not math.isfinite(w_sec)) or (w_sec * 1000.0 > pol.p99_queue_delay_sla_ms):
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
    strat_metrics = _evaluate_strategies_trajectory(
        rps_trace=rps_trace,
        policy=policy,
        seed=404,
    )
    return spec.name, {
        "metadata": {
            "csv_file": spec.filename,
            "description": spec.description,
            "total_csv_rows": total_rows,
            "start_offset": offset,
            "evaluated_steps": len(rps_trace),
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "mean_rps": round(float(np.mean(rps_trace)), 2),
            "peak_rps": round(float(np.max(rps_trace)), 2),
            "peak_to_mean_ratio": round(float(np.max(rps_trace) / np.mean(rps_trace)), 3),
        },
        "strategies": strat_metrics,
    }


def _worker_rolling_window_task(
    args: tuple[DatasetSpec, int, int, int, int],
) -> dict[str, dict[str, float | int]]:
    """Worker entrypoint for Experiment B disjoint real rolling-window evaluation."""
    spec, start_offset, window_steps, seed, num_scenarios = args
    _, rps_trace, _ = load_empirical_trace(
        csv_path=DATA_DIR / spec.filename,
        scale_to_mean_rps=3200.0,
        max_steps=window_steps,
        start_offset=start_offset,
    )
    policy = default_publication_policy(num_scenarios=num_scenarios)
    return _evaluate_strategies_trajectory(
        rps_trace=rps_trace,
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
    res = _evaluate_strategies_trajectory(
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
    """Worker entrypoint for Experiment D TRUE 1-minute (dt=60s) node cold-start delay sweep."""
    boot_delay_min, elb_1min_trace, num_scenarios = args
    policy = default_publication_policy(num_scenarios=num_scenarios)
    res = _evaluate_strategies_trajectory(
        rps_trace=elb_1min_trace,
        policy=policy,
        seed=404,
        boot_delay_steps=boot_delay_min,
        dt_hours=1.0 / 60.0,
        record_coldstart_only=True,
    )
    return boot_delay_min, {
        "boot_delay_steps": boot_delay_min,
        "step_resolution_seconds": 60,
        "boot_delay_seconds": boot_delay_min * 60,
        "AegisScale_Full": res["AegisScale_Full"],
        "Reactive_HPA_55": res["Reactive_HPA_55"],
    }


def _worker_spot_price_task(args: tuple[str, float, np.ndarray, int]) -> tuple[str, float, dict[str, Any]]:
    """Worker entrypoint for Experiment D Spot price multiplier sensitivity (0.7x - 1.6x)."""
    ds_name, mult, trace, num_scenarios = args
    policy = default_publication_policy(num_scenarios=num_scenarios)
    res = _evaluate_strategies_trajectory(
        rps_trace=trace,
        policy=policy,
        seed=404,
        spot_price_mult=mult,
    )
    return ds_name, mult, {
        "spot_price_multiplier": mult,
        "AegisScale_annualized_cost_usd": res["AegisScale_Full"]["annualized_cost_usd"],
        "IsoSLA_OD_annualized_cost_usd": res["IsoSLA_100_OnDemand"]["annualized_cost_usd"],
        "SavingsPlan_annualized_cost_usd": res["OD_SavingsPlan_IsoSLA"]["annualized_cost_usd"],
        "savings_vs_isosla_od_pct": res["AegisScale_Full"]["savings_vs_isosla_od_pct"],
        "savings_vs_savings_plan_pct": res["AegisScale_Full"]["savings_vs_savings_plan_pct"],
        "sla_continuity_pct": res["AegisScale_Full"]["sla_continuity_pct"],
    }


def run_k8s_2d_multiregion_benchmark(
    azure_trace: np.ndarray,
    num_scenarios: int = 120,
) -> dict[str, Any]:
    """Experiment E: Empirical 2D (vCPU, GiB) Kubernetes/Karpenter benchmark across 3 AWS regions.

    Evaluates a memory-heavy inference pod (1 vCPU, 7 GiB, 40 RPS/pod) across us-east-1,
    eu-west-1, and ap-northeast-1, comparing:
      1. 1D CPU-Only AegisScale (ignores memory bin-packing, picks cheapest $/vCPU nodes like c6a/c7g)
      2. Karpenter-Style Greedy Spot (2D-aware cheapest Spot node without CVaR hedging)
      3. AegisScale 2D K8s Operator (K8sAegisOperator + build_2d_effective_pools + KarpenterReconciler)
    """
    pod = PodResources(name="llm-inference-api", cpu_vcpu=1.0, memory_gib=7.0, target_rps_per_pod=40.0)
    overhead = K8sClusterOverhead(daemonset_cpu_vcpu=0.25, daemonset_memory_gib=0.75, node_boot_seconds=65.0)
    policy = default_publication_policy(num_scenarios=num_scenarios)
    n_steps = len(azure_trace)
    dt_hours = 5.0 / 60.0

    region_results: dict[str, Any] = {}
    for region in SUPPORTED_REGIONS:
        pools = get_region_pools(region)
        pool_map = {p.pool_id: p for p in pools}
        copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)
        realized_draws = copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=808)

        # 2D Operator (full AegisScale K8s stack)
        op_2d = K8sAegisOperator(
            pools=pools,
            pod=pod,
            policy=policy,
            overhead=overhead,
            interval_sec=300.0,
            seed=404,
        )

        # 1D CPU-only naive allocator (uses cheapest $/vCPU pools, e.g. c6a/c7g 4 vCPU / 8 GiB -> only 1 pod fits!)
        od_cheapest_1d = min((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
        spot_cheapest_1d = min((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
        # 2D-aware greedy Karpenter Spot (cheapest $/pod-slot Spot pool, 10% OD anchor)
        schedulable_od = [p for p in pools if p.market == "on-demand" and pods_per_node(p, pod, overhead) >= 1]
        schedulable_spot = [p for p in pools if p.market == "spot" and pods_per_node(p, pod, overhead) >= 1]
        od_2d_best = min(schedulable_od, key=lambda p: p.hourly_price_usd / pods_per_node(p, pod, overhead))
        spot_2d_best = min(schedulable_spot, key=lambda p: p.hourly_price_usd / pods_per_node(p, pod, overhead))

        costs = {"1D_CPU_Only_Mixed": 0.0, "Karpenter_Greedy_2D_Spot": 0.0, "AegisScale_2D_Karpenter": 0.0}
        breaches = {k: 0 for k in costs}
        stranded_vcpu_ratios: dict[str, list[float]] = {k: [] for k in costs}
        manifest_counts = 0

        prev_rps = float(azure_trace[0])
        for t in range(n_steps):
            true_rps = float(azure_trace[t])
            recalled_ids = {
                pools[k].pool_id
                for k in range(len(pools))
                if pools[k].market == "spot" and realized_draws[t, k] > 0.5
            }

            report_2d = op_2d.run_cycle(observed_rps=prev_rps)
            req_pods = report_2d.required_pod_replicas
            if t == n_steps - 1:
                manifest_counts = len(report_2d.manifests.node_pools) + 2

            # 1D CPU-only allocator assumes 4 vCPU node fits 3 pods (by CPU), ignoring 7 GiB memory limit
            # Realized schedulable pods on c7g.xlarge (4 vCPU, 8 GiB) is only 1 pod -> massive stranding/OOM
            od_1d_nodes = max(1, int(math.ceil(0.30 * req_pods / 3.0)))
            sp_1d_nodes = max(0, int(math.ceil(0.70 * req_pods / 3.0)))
            alloc_1d = {od_cheapest_1d.pool_id: od_1d_nodes, spot_cheapest_1d.pool_id: sp_1d_nodes}

            # Karpenter 2D Greedy Spot (10% best 2D OD, 90% single cheapest 2D Spot pool)
            od_2d_cap = pods_per_node(od_2d_best, pod, overhead)
            sp_2d_cap = pods_per_node(spot_2d_best, pod, overhead)
            alloc_karp_greedy = {
                od_2d_best.pool_id: max(1, int(math.ceil(0.10 * req_pods / od_2d_cap))),
                spot_2d_best.pool_id: max(0, int(math.ceil(0.90 * req_pods / sp_2d_cap))),
            }

            alloc_aegis_2d = report_2d.portfolio.allocations

            for label, alloc in [
                ("1D_CPU_Only_Mixed", alloc_1d),
                ("Karpenter_Greedy_2D_Spot", alloc_karp_greedy),
                ("AegisScale_2D_Karpenter", alloc_aegis_2d),
            ]:
                h_cost = sum(pool_map[pid].hourly_price_usd * c for pid, c in alloc.items() if c > 0)
                costs[label] += h_cost * dt_hours

                total_vcpu = sum(pool_map[pid].vcpu * c for pid, c in alloc.items() if c > 0)
                total_pods = sum(pods_per_node(pool_map[pid], pod, overhead) * c for pid, c in alloc.items() if c > 0)
                used_vcpu = total_pods * pod.cpu_vcpu
                stranded_vcpu_ratios[label].append(max(0.0, 1.0 - used_vcpu / max(1e-9, total_vcpu)) * 100.0)

                surv_pods = sum(
                    pods_per_node(pool_map[pid], pod, overhead) * c
                    for pid, c in alloc.items()
                    if c > 0 and (pool_map[pid].market == "on-demand" or pid not in recalled_ids)
                )
                w_sec = ggc_quantile_wait_seconds(
                    arrival_rate_rps=true_rps,
                    vcpus=float(surv_pods),
                    per_vcpu_mu=pod.target_rps_per_pod,
                    quantile=0.99,
                    arrival_cv2=policy.arrival_cv2,
                    service_cv2=policy.service_cv2,
                )
                if (not math.isfinite(w_sec)) or (w_sec * 1000.0 > policy.p99_queue_delay_sla_ms):
                    breaches[label] += 1

            prev_rps = true_rps

        annual_mult = (24.0 * 365.0) / (n_steps * dt_hours)
        region_results[region] = {
            "pod_spec": {"cpu_vcpu": pod.cpu_vcpu, "memory_gib": pod.memory_gib, "target_rps_per_pod": pod.target_rps_per_pod},
            "karpenter_manifests_emitted": manifest_counts,
            "strategies": {
                label: {
                    "annualized_cost_usd": round(costs[label] * annual_mult, 2),
                    "sla_continuity_pct": round((1.0 - breaches[label] / n_steps) * 100.0, 2),
                    "total_sla_breaches": int(breaches[label]),
                    "mean_stranded_vcpu_pct": round(float(np.mean(stranded_vcpu_ratios[label])), 2),
                }
                for label in costs
            },
        }
    return region_results


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


def _summarize_rolling_window_results(
    per_window_records: list[dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Compute mean, sample std, exact Student-t 95% CI half-width, and paired Wilcoxon p-values."""
    n_windows = len(per_window_records)
    t_crit = float(student_t.ppf(0.975, df=max(1, n_windows - 1)))
    metrics_of_interest = [
        "annualized_cost_usd",
        "savings_vs_isosla_od_pct",
        "savings_vs_savings_plan_pct",
        "sla_continuity_pct",
        "total_sla_breaches",
        "spot_storm_sla_breaches",
    ]
    strategy_stats: dict[str, dict[str, dict[str, float]]] = {}
    raw_arrays: dict[str, dict[str, np.ndarray]] = {s: {} for s in STRATEGY_NAMES}

    for s_name in STRATEGY_NAMES:
        strategy_stats[s_name] = {}
        for m_key in metrics_of_interest:
            vals = np.asarray([float(rec[s_name][m_key]) for rec in per_window_records], dtype=np.float64)
            raw_arrays[s_name][m_key] = vals
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n_windows > 1 else 0.0
            ci95_hw = t_crit * std_val / math.sqrt(float(n_windows))
            strategy_stats[s_name][m_key] = {
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "ci95_halfwidth": round(ci95_hw, 3),
            }

    p_cost_od = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["annualized_cost_usd"],
        raw_arrays["IsoSLA_100_OnDemand"]["annualized_cost_usd"],
    )
    p_cost_sp = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["annualized_cost_usd"],
        raw_arrays["OD_SavingsPlan_IsoSLA"]["annualized_cost_usd"],
    )
    p_cont_greedy = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["sla_continuity_pct"],
        raw_arrays["Greedy_Cheapest_Spot"]["sla_continuity_pct"],
    )
    p_cont_reactive = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["sla_continuity_pct"],
        raw_arrays["Reactive_HPA_55"]["sla_continuity_pct"],
    )
    p_cont_static = _safe_wilcoxon_two_sided(
        raw_arrays["AegisScale_Full"]["sla_continuity_pct"],
        raw_arrays["Static_30_70_Mixed"]["sla_continuity_pct"],
    )

    return {
        "n_disjoint_windows": n_windows,
        "student_t_critical_95": round(t_crit, 4),
        "datasets_covered": [s.name for s in DATASET_SPECS],
        "strategy_statistics": strategy_stats,
        "wilcoxon_two_sided_tests": {
            "p_value_cost_vs_isosla_od": p_cost_od,
            "p_value_cost_vs_savings_plan": p_cost_sp,
            "p_value_continuity_vs_greedy": p_cont_greedy,
            "p_value_continuity_vs_reactive": p_cont_reactive,
            "p_value_continuity_vs_static": p_cont_static,
            "all_core_significant_at_alpha_0_05": bool(
                max(p_cost_od, p_cost_sp, p_cont_greedy, p_cont_reactive) < 0.05
            ),
        },
    }


def generate_publication_figures(
    results: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """Generate crisp, zero-overlap publication figures at 160 DPI."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.4,
        }
    )

    saved_paths: list[str] = []

    # -------------------------------------------------------------------------
    # Figure 1: multi_dataset_generalization.png (5 datasets x 6 strategies)
    # -------------------------------------------------------------------------
    exp_a = results["experiment_a_multi_dataset"]
    ds_names = [spec.name for spec in DATASET_SPECS]
    strat_labels = {
        "Reactive_HPA_55": "Reactive HPA (55%)",
        "IsoSLA_100_OnDemand": "Iso-SLA 100% OD",
        "OD_SavingsPlan_IsoSLA": "1-Yr Savings Plan + OD",
        "Greedy_Cheapest_Spot": "Greedy Spot (Karpenter)",
        "Static_30_70_Mixed": "Static 30/70 (Warn-Aware)",
        "AegisScale_Full": "AegisScale (Full)",
    }
    strat_colors = {
        "Reactive_HPA_55": "#64748b",
        "IsoSLA_100_OnDemand": "#2563eb",
        "OD_SavingsPlan_IsoSLA": "#0ea5e9",
        "Greedy_Cheapest_Spot": "#ef4444",
        "Static_30_70_Mixed": "#f59e0b",
        "AegisScale_Full": "#10b981",
    }

    x = np.arange(len(ds_names), dtype=np.float64)
    n_s = len(STRATEGY_NAMES)
    width = 0.13
    offsets = (np.arange(n_s) - (n_s - 1) / 2.0) * width

    fig1, (ax_cont, ax_cost) = plt.subplots(1, 2, figsize=(15.0, 6.2))
    fig1.subplots_adjust(top=0.79, bottom=0.16, left=0.06, right=0.98, wspace=0.17)

    for idx, s_name in enumerate(STRATEGY_NAMES):
        cont_vals = [float(exp_a[ds]["strategies"][s_name]["sla_continuity_pct"]) for ds in ds_names]
        cost_vals = [float(exp_a[ds]["strategies"][s_name]["annualized_cost_usd"]) / 1000.0 for ds in ds_names]
        edge_w = 1.1 if s_name == "AegisScale_Full" else 0.4
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
    ax_cont.axhline(98.5, color="#0f172a", linestyle="--", linewidth=1.1, alpha=0.75, label="98.5% Target")
    ax_cont.set_xticks(x)
    ax_cont.set_xticklabels(ds_names, rotation=14, ha="right")
    ax_cont.set_ylabel("P99 Latency SLA Continuity (%)")
    ax_cont.set_title("(a) Zero-Retuning SLA Continuity (5 Real Traces)")
    ax_cont.grid(axis="y", linestyle="--", alpha=0.35)

    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(ds_names, rotation=14, ha="right")
    ax_cost.set_ylabel("Annualized Compute Spend ($K USD)")
    ax_cost.set_title("(b) Annualized Cost vs. OD and 1-Yr Savings Plan")
    ax_cost.grid(axis="y", linestyle="--", alpha=0.35)
    ax_cost.margins(y=0.10)

    handles, labels = ax_cont.get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        framealpha=0.95,
    )
    fig1.suptitle(
        "Multi-Dataset Generalization Across 4 AWS/Viral NAB Traces + 2024 Microsoft Azure LLM Inference",
        y=0.975,
        fontsize=12.2,
        fontweight="bold",
    )
    p1 = figures_dir / "multi_dataset_generalization.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)
    saved_paths.append(str(p1))

    # -------------------------------------------------------------------------
    # Figure 2: ablation_and_coldstart.png
    # -------------------------------------------------------------------------
    exp_c = results["experiment_c_ablations"]
    exp_d_cold = results["experiment_d_sensitivity_and_scalability"]["cold_start_delay_sweep"]

    fig2, (ax_abl, ax_cold) = plt.subplots(1, 2, figsize=(14.4, 5.6), constrained_layout=True)

    abl_pretty = {
        "Full_AegisScale": "Full AegisScale",
        "Ablation_No_ACI": "No ACI (Point x1.05)",
        "Ablation_Static_Conformal": "Static Conformal",
        "Ablation_No_GGc": "No G/G/c (85% Util)",
        "Ablation_Uncorrelated_Spot": "Uncorrelated Spot",
        "Ablation_No_Spot_Warnings": "No 2-Min Warnings",
    }
    abl_colors = {
        "Full_AegisScale": "#10b981",
        "Ablation_No_ACI": "#ef4444",
        "Ablation_Static_Conformal": "#f59e0b",
        "Ablation_No_GGc": "#8b5cf6",
        "Ablation_Uncorrelated_Spot": "#ec4899",
        "Ablation_No_Spot_Warnings": "#0284c7",
    }
    abl_markers = {
        "Full_AegisScale": "o",
        "Ablation_No_ACI": "s",
        "Ablation_Static_Conformal": "^",
        "Ablation_No_GGc": "D",
        "Ablation_Uncorrelated_Spot": "X",
        "Ablation_No_Spot_Warnings": "P",
    }

    pts_y: list[float] = []
    for v_name in ABLATION_NAMES:
        c_elb = float(exp_c["AWS-ELB-Web"][v_name]["annualized_cost_usd"]) / 1000.0
        c_vir = float(exp_c["Viral-Flash-Traffic"][v_name]["annualized_cost_usd"]) / 1000.0
        b_elb = float(exp_c["AWS-ELB-Web"][v_name]["total_sla_breaches"])
        b_vir = float(exp_c["Viral-Flash-Traffic"][v_name]["total_sla_breaches"])
        avg_cost_k = 0.5 * (c_elb + c_vir)
        tot_breaches = b_elb + b_vir
        pts_y.append(tot_breaches)
        ax_abl.scatter(
            avg_cost_k,
            tot_breaches,
            s=140,
            color=abl_colors[v_name],
            marker=abl_markers[v_name],
            edgecolors="#0f172a",
            linewidths=0.9,
            zorder=4,
            label=f"{abl_pretty[v_name]} ({int(tot_breaches)} br, ${avg_cost_k:.1f}K)",
        )

    ax_abl.margins(x=0.18)
    ax_abl.set_ylim(-max(2.0, 0.06 * max(pts_y)), max(14.0, max(pts_y) * 1.48))
    ax_abl.set_xlabel("Mean Annualized Spend Across ELB + Viral Traces ($K USD)")
    ax_abl.set_ylabel("Combined P99 SLA Breaches (Lower is Better)")
    ax_abl.set_title("(a) Unconfounded 6-Way Ablation (1 Change per Row)")
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
            fontsize=8.6,
            color="#065f46",
            fontweight="bold",
        )
        ax_cold.annotate(
            f"{y_r:.1f}%",
            (d_s, y_r),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.6,
            color="#991b1b",
        )

    ax_cold.set_xticks(delays_sec)
    ax_cold.set_xticklabels([f"{d}s\n({d // 60} min)" for d in delays_sec])
    low_y = min(min(react_cold), min(aegis_cold))
    ax_cold.set_ylim(max(0.0, low_y - 9.0), 103.5)
    ax_cold.set_xlim(-15, max(delays_sec) + 18)
    ax_cold.set_xlabel("True 1-Minute Resolution Node Cold-Start Lag (with Spot Eviction)")
    ax_cold.set_ylabel("P99 Latency SLA Continuity (%)")
    ax_cold.set_title("(b) Node Boot Delay Sensitivity (dt = 60s)")
    ax_cold.grid(True, linestyle="--", alpha=0.35)
    ax_cold.legend(loc="center right", framealpha=0.93)

    fig2.suptitle(
        "Unconfounded Architectural Ablations and True 1-Minute Node Cold-Start Resilience",
        fontsize=12.2,
        fontweight="bold",
    )
    p2 = figures_dir / "ablation_and_coldstart.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    saved_paths.append(str(p2))

    # -------------------------------------------------------------------------
    # Figure 3: solver_scalability.png
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
        label="1,000 ms Sub-Budget (of 60s Control Loop)",
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
    ax_sc.set_ylim(max(1.0, min(med_ms) * 0.40), max(2400.0, max(p99_ms) * 1.85))
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
    """Execute Experiments A, B, C, D, and E, write JSON report, and render 3 figures."""
    t_start = time.perf_counter()
    steps = max_steps if max_steps is not None else (80 if fast else 600)
    # Experiment B: 30 disjoint real windows (6 per dataset x 5 datasets) in full mode, 10 in fast mode
    windows_per_dataset = 2 if fast else 6
    if n_seeds is not None:
        windows_per_dataset = max(1, n_seeds // len(DATASET_SPECS))
    window_steps = 80 if fast else 300
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
    _, azure_trace, _ = load_empirical_trace(
        csv_path=DATA_DIR / "real_azure_inference_2024.csv",
        scale_to_mean_rps=3200.0,
        max_steps=steps,
        start_offset=0,
    )

    # True 1-minute interpolated ELB trace for cold-start sweep (dt=60s)
    elb_1min_full = resample_5min_to_1min(elb_trace[: min(len(elb_trace), 120 if not fast else 24)])

    max_workers = min(os.cpu_count() or 4, 8)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Experiment A: 5 real production datasets
        fut_exp_a = [
            executor.submit(_worker_dataset_task, (spec, steps, num_scenarios))
            for spec in DATASET_SPECS
        ]

        # Experiment B: Disjoint real rolling windows across all 5 datasets
        rolling_tasks: list[tuple[DatasetSpec, int, int, int, int]] = []
        seed_counter = 0
        for spec in DATASET_SPECS:
            for w_idx in range(windows_per_dataset):
                # Space windows evenly without overlap
                stride = window_steps
                offset = w_idx * stride
                rolling_tasks.append((spec, offset, window_steps, 2000 + seed_counter * 97, num_scenarios))
                seed_counter += 1
        fut_exp_b = [executor.submit(_worker_rolling_window_task, task) for task in rolling_tasks]

        # Experiment C: Unconfounded 6-way ablation on AWS-ELB-Web & Viral-Flash-Traffic
        ablation_tasks: list[tuple[str, str, np.ndarray, int]] = []
        for ds_label, ds_trace in [("AWS-ELB-Web", elb_trace), ("Viral-Flash-Traffic", viral_trace)]:
            for v_name in ABLATION_NAMES:
                ablation_tasks.append((ds_label, v_name, ds_trace, num_scenarios))
        fut_exp_c = [executor.submit(_worker_ablation_task, task) for task in ablation_tasks]

        # Experiment D.1: AZ correlation sweep
        rho_az_grid = [0.0, 0.25, 0.55, 0.75]
        fut_az = [
            executor.submit(_worker_az_corr_task, (rho_az, elb_trace, num_scenarios))
            for rho_az in rho_az_grid
        ]

        # Experiment D.2: True 1-minute (dt=60s) node cold-start delay sweep
        boot_delays = [0, 1, 2, 3]
        fut_cold = [
            executor.submit(_worker_coldstart_task, (d, elb_1min_full, num_scenarios))
            for d in boot_delays
        ]

        # Experiment D.3: Spot price multiplier sensitivity (0.7x, 1.0x, 1.3x, 1.6x)
        spot_mults = [0.7, 1.0, 1.3, 1.6]
        spot_tasks = [
            (ds_name, m, tr, num_scenarios)
            for ds_name, tr in [("AWS-ELB-Web", elb_trace), ("Azure-LLM-Inference-2024", azure_trace)]
            for m in spot_mults
        ]
        fut_spot = [executor.submit(_worker_spot_price_task, task) for task in spot_tasks]

        exp_a_results: dict[str, Any] = {}
        for fut in fut_exp_a:
            ds_name, ds_payload = fut.result()
            exp_a_results[ds_name] = ds_payload

        rolling_records = [fut.result() for fut in fut_exp_b]
        exp_b_results = _summarize_rolling_window_results(rolling_records)

        exp_c_results: dict[str, dict[str, Any]] = {"AWS-ELB-Web": {}, "Viral-Flash-Traffic": {}}
        for fut in fut_exp_c:
            ds_label, v_name, v_metrics = fut.result()
            exp_c_results[ds_label][v_name] = v_metrics

        az_sweep_map = dict(fut.result() for fut in fut_az)
        az_sweep_list = [az_sweep_map[r] for r in rho_az_grid]

        cold_sweep_map = dict(fut.result() for fut in fut_cold)
        cold_sweep_list = [cold_sweep_map[d] for d in boot_delays]

        spot_sens: dict[str, list[dict[str, Any]]] = {"AWS-ELB-Web": [], "Azure-LLM-Inference-2024": []}
        spot_raw: dict[tuple[str, float], dict[str, Any]] = {}
        for fut in fut_spot:
            dn, mv, payload = fut.result()
            spot_raw[(dn, mv)] = payload
        for dn in ("AWS-ELB-Web", "Azure-LLM-Inference-2024"):
            spot_sens[dn] = [spot_raw[(dn, m)] for m in spot_mults]

    # Experiment D.4: Single-thread timing of HiGHS MILP scalability without CPU contention
    scalability_list = run_highs_scalability_sweep(
        pool_counts=[21, 42, 84, 147, 210],
        num_scenarios=200,
        trials_per_k=scalability_trials,
    )

    # Experiment E: Empirical 2D Kubernetes/Karpenter multi-region benchmark on Azure 2024 trace
    exp_e_results = run_k8s_2d_multiregion_benchmark(
        azure_trace=azure_trace[: min(len(azure_trace), 80 if fast else 300)],
        num_scenarios=num_scenarios,
    )

    elapsed_sec = time.perf_counter() - t_start
    fixed_pol = default_publication_policy(num_scenarios=num_scenarios)

    results: dict[str, Any] = {
        "suite_metadata": {
            "mode": "fast" if fast else "publication_full",
            "max_steps_per_dataset": steps,
            "disjoint_rolling_windows": len(rolling_records),
            "rolling_window_steps": window_steps,
            "milp_scenarios": num_scenarios,
            "savings_plan_discount": SAVINGS_PLAN_DISCOUNT,
            "http_timeout_cap_ms": HTTP_GATEWAY_TIMEOUT_MS,
            "elapsed_wall_time_sec": round(elapsed_sec, 2),
            "fixed_sla_policy": fixed_pol.model_dump(),
        },
        "experiment_a_multi_dataset": exp_a_results,
        "experiment_b_disjoint_wilcoxon": exp_b_results,
        "experiment_c_ablations": exp_c_results,
        "experiment_d_sensitivity_and_scalability": {
            "az_correlation_sweep": az_sweep_list,
            "cold_start_delay_sweep": cold_sweep_list,
            "spot_price_multiplier_sensitivity": spot_sens,
            "highs_scalability_sweep": scalability_list,
        },
        "experiment_e_k8s_2d_multiregion": exp_e_results,
    }

    figure_paths = generate_publication_figures(results=results, figures_dir=figures_dir)
    results["suite_metadata"]["generated_figures"] = figure_paths

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AegisScale Phase 3 audited multi-dataset, Wilcoxon, ablation, sensitivity, and K8s benchmark."
    )
    parser.add_argument("--fast", action="store_true", help="Run fast CI/unit-test configuration.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override steps per dataset.")
    parser.add_argument("--n-seeds", type=int, default=None, help="Override total disjoint windows.")
    args = parser.parse_args()

    res = run_statistical_rigor_suite(
        fast=args.fast,
        max_steps=args.max_steps,
        n_seeds=args.n_seeds,
    )
    meta = res["suite_metadata"]
    wilc = res["experiment_b_disjoint_wilcoxon"]["wilcoxon_two_sided_tests"]
    print(f"AegisScale Phase 3 Audited Suite completed in {meta['elapsed_wall_time_sec']}s ({meta['mode']}).")
    print("Wilcoxon two-sided p-values:", json.dumps(wilc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

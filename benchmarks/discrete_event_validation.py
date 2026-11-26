"""Independent Multi-Server G/G/c Discrete-Event Queue Simulator (DES) Validation.

Validates AegisScale's realized P99 tail queueing latency and SLA continuity using
an exact event-by-event FCFS multi-server G/G/c discrete-event simulator (Python
heapq min-heap of server-available timestamps), 100% independent of the analytical
Kingman-Sakasegawa heavy-traffic approximation.

Part 1 (Formula Calibration Check):
    Compares Kingman-Sakasegawa analytical P99 wait vs. empirical DES P99 wait
    (100,000 simulated requests per point after steady-state warmup) across
    c in [8, 32] servers and utilization rho in [0.55, 0.65, 0.75, 0.85, 0.90, 0.95]
    with Gamma inter-arrival times (cv_a^2 = 1.6) and Gamma service times
    (cv_s^2 = 1.2, mean = 1/125 s = 8.0 ms).

Part 2 (End-to-End Trace Replay with DES Oracle):
    Replays 200-step windows of AWS-ELB-Web (real_aws_elb_cloudwatch.csv) and
    Azure-LMM-2024 (real_azure_inference_2024.csv) using the exact same causal
    prefix normalization and exogenous Spot storms as statistical_rigor.py,
    evaluating Reactive_HPA_55, Seasonal_Predictive_OD, Greedy_Cheapest_Spot,
    and AegisScale_Full with true request-level DES simulation carrying backlog
    smoothly across intervals.

Usage:
    .venv\\Scripts\\python.exe benchmarks/discrete_event_validation.py
    .venv\\Scripts\\python.exe benchmarks/discrete_event_validation.py --fast
"""

from __future__ import annotations

import argparse
import concurrent.futures
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

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
from aegis_scale.math_core.queuing import (
    ggc_mean_wait_seconds,
    ggc_quantile_wait_seconds,
    ggc_wait_probability,
    required_safe_vcpus,
)
from benchmarks.statistical_rigor import (
    DATA_DIR,
    HTTP_GATEWAY_TIMEOUT_MS,
    _build_exogenous_storm_schedule,
    _scale_spot_prices,
    default_publication_policy,
    load_empirical_trace,
)

DEFAULT_OUTPUT_JSON = ROOT / "output" / "discrete_event_validation.json"
DEFAULT_FIGURE_PATH = ROOT / "docs" / "figures" / "des_validation.png"

DES_STRATEGIES = [
    "Reactive_HPA_55",
    "Seasonal_Predictive_OD",
    "Greedy_Cheapest_Spot",
    "AegisScale_Full",
]

TRACE_SPECS = [
    (
        "AWS-ELB-Web",
        "real_aws_elb_cloudwatch.csv",
        "AWS CloudWatch Elastic Load Balancer HTTP RequestCount (NAB realAWSCloudwatch)",
    ),
    (
        "Azure-LMM-2024",
        "real_azure_inference_2024.csv",
        "Microsoft Azure Public Dataset Oct 2024 Multimodal LLM Inference (5-min bins)",
    ),
]


def _sample_gamma_positive(
    rng: np.random.Generator,
    mean_sec: float,
    cv2: float,
    size: int,
) -> np.ndarray:
    """Sample positive i.i.d. Gamma variates with exact mean and squared CV."""
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    if cv2 <= 0.0:
        return np.full(size, mean_sec, dtype=np.float64)
    shape = 1.0 / cv2
    scale = mean_sec * cv2
    return rng.gamma(shape=shape, scale=scale, size=size)


def sample_interval_requests(
    rng: np.random.Generator,
    arrival_rate_rps: float,
    interval_sec: float,
    per_vcpu_mu: float = 125.0,
    arrival_cv2: float = 1.6,
    service_cv2: float = 1.2,
) -> tuple[list[float], list[float]]:
    """Sample exact renewal arrival times in (0, interval_sec] and service times.

    Returns Python lists of arrival timestamps (seconds from interval start) and
    corresponding Gamma service times (seconds) for fast heapq simulation.
    """
    if arrival_rate_rps <= 0.0 or interval_sec <= 0.0:
        return [], []

    mean_ia = 1.0 / arrival_rate_rps
    mean_st = 1.0 / per_vcpu_mu

    # Over-allocate slightly so one vectorized Gamma draw covers the entire interval.
    expected_n = max(64, int(math.ceil(arrival_rate_rps * interval_sec * 1.18)) + 64)
    chunks: list[np.ndarray] = []
    total_time = 0.0
    while total_time <= interval_sec:
        ia_chunk = _sample_gamma_positive(rng, mean_ia, arrival_cv2, expected_n)
        chunks.append(ia_chunk)
        total_time += float(np.sum(ia_chunk))
        expected_n = max(64, expected_n // 2)

    all_ia = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    arr_times = np.cumsum(all_ia)
    cutoff = int(np.searchsorted(arr_times, interval_sec, side="right"))
    arr_slice = arr_times[:cutoff]
    st_slice = _sample_gamma_positive(rng, mean_st, service_cv2, cutoff)
    return arr_slice.tolist(), st_slice.tolist()


class GGcDiscreteEventSimulator:
    """Exact event-by-event FCFS multi-server G/G/c discrete-event simulator.

    Uses a Python heapq min-heap of c server-available timestamps (Kiefer-Wolfowitz
    recursion). Supports both stationary fixed-count simulation (Part 1) and
    multi-interval trace replay with smooth server-count resizing and backlog
    carryover (Part 2).
    """

    def __init__(
        self,
        num_servers: int,
        max_backlog_sec: float = HTTP_GATEWAY_TIMEOUT_MS / 1000.0,
    ) -> None:
        self.max_backlog_sec = float(max_backlog_sec)
        c = max(0, int(num_servers))
        self.server_heap: list[float] = [0.0] * c

    def resize_servers(self, new_num_servers: int) -> None:
        """Adjust active server count at an interval boundary while preserving backlog."""
        c_new = max(0, int(new_num_servers))
        c_old = len(self.server_heap)
        if c_new == c_old:
            return
        if c_new == 0:
            self.server_heap = []
            return
        if c_new > c_old:
            self.server_heap.extend([0.0] * (c_new - c_old))
            heapq.heapify(self.server_heap)
            return
        # Scale down or Spot preemption: preserve the c_new heaviest backlog tails.
        kept = sorted(self.server_heap, reverse=True)[:c_new]
        heapq.heapify(kept)
        self.server_heap = kept

    def run_interval(
        self,
        arrival_times: list[float],
        service_times: list[float],
        interval_sec: float,
        num_servers: int,
    ) -> dict[str, float | int]:
        """Simulate all requests arriving in [0, interval_sec] and carry backlog forward."""
        n_req = len(arrival_times)
        self.resize_servers(num_servers)

        if num_servers <= 0:
            return {
                "requests": n_req,
                "p99_wait_ms": HTTP_GATEWAY_TIMEOUT_MS,
                "mean_wait_ms": HTTP_GATEWAY_TIMEOUT_MS,
                "p_wait": 1.0,
                "carried_backlog_ms": HTTP_GATEWAY_TIMEOUT_MS,
            }

        if n_req == 0:
            shifted = [max(0.0, s - interval_sec) for s in self.server_heap]
            heapq.heapify(shifted)
            self.server_heap = shifted
            max_rem_ms = (max(shifted) * 1000.0) if shifted else 0.0
            return {
                "requests": 0,
                "p99_wait_ms": 0.0,
                "mean_wait_ms": 0.0,
                "p_wait": 0.0,
                "carried_backlog_ms": max_rem_ms,
            }

        heap = self.server_heap
        heapreplace = heapq.heapreplace
        waits = [0.0] * n_req
        wait_count = 0

        for idx in range(n_req):
            t_arr = arrival_times[idx]
            st = service_times[idx]
            earliest = heap[0]
            if earliest > t_arr:
                w = earliest - t_arr
                waits[idx] = w
                wait_count += 1
                heapreplace(heap, earliest + st)
            else:
                heapreplace(heap, t_arr + st)

        # Shift server-free timestamps to the start of the next interval (t = 0.0),
        # capping runaway overload backlog at the HTTP gateway timeout (5.0s).
        cap_sec = self.max_backlog_sec
        shifted = [min(cap_sec, max(0.0, s - interval_sec)) for s in heap]
        heapq.heapify(shifted)
        self.server_heap = shifted

        waits_arr = np.asarray(waits, dtype=np.float64)
        p99_ms = min(HTTP_GATEWAY_TIMEOUT_MS, float(np.quantile(waits_arr, 0.99)) * 1000.0)
        mean_ms = min(HTTP_GATEWAY_TIMEOUT_MS, float(np.mean(waits_arr)) * 1000.0)
        p_wait = float(wait_count) / float(n_req)
        carried_ms = (max(shifted) * 1000.0) if shifted else 0.0

        return {
            "requests": n_req,
            "p99_wait_ms": p99_ms,
            "mean_wait_ms": mean_ms,
            "p_wait": p_wait,
            "carried_backlog_ms": carried_ms,
        }


def simulate_stationary_ggc(
    num_servers: int,
    rho: float,
    per_vcpu_mu: float = 125.0,
    arrival_cv2: float = 1.6,
    service_cv2: float = 1.2,
    num_requests: int = 100_000,
    warmup_requests: int = 15_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Run an exact stationary G/G/c DES for (warmup + num_requests) arrivals.

    Discards the initial warmup_requests to eliminate empty-system transient bias,
    then computes empirical P99 wait, mean wait, and P(W_q > 0) over num_requests.
    """
    if num_servers < 1:
        raise ValueError("num_servers must be >= 1")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be strictly in (0, 1) for stationary simulation")

    rng = np.random.default_rng(seed)
    arrival_rate_rps = rho * float(num_servers) * per_vcpu_mu
    mean_ia = 1.0 / arrival_rate_rps
    mean_st = 1.0 / per_vcpu_mu

    total_n = warmup_requests + num_requests
    ia_list = _sample_gamma_positive(rng, mean_ia, arrival_cv2, total_n).tolist()
    st_list = _sample_gamma_positive(rng, mean_st, service_cv2, total_n).tolist()

    heap = [0.0] * num_servers
    heapreplace = heapq.heapreplace
    t_now = 0.0

    # Warmup phase (not recorded)
    for i in range(warmup_requests):
        t_now += ia_list[i]
        earliest = heap[0]
        if earliest > t_now:
            heapreplace(heap, earliest + st_list[i])
        else:
            heapreplace(heap, t_now + st_list[i])

    # Measurement phase
    waits = [0.0] * num_requests
    wait_count = 0
    for j in range(num_requests):
        idx = warmup_requests + j
        t_now += ia_list[idx]
        earliest = heap[0]
        if earliest > t_now:
            w = earliest - t_now
            waits[j] = w
            wait_count += 1
            heapreplace(heap, earliest + st_list[idx])
        else:
            heapreplace(heap, t_now + st_list[idx])

    waits_arr = np.asarray(waits, dtype=np.float64)
    return {
        "num_servers": int(num_servers),
        "rho": float(rho),
        "arrival_rate_rps": round(arrival_rate_rps, 3),
        "measured_requests": int(num_requests),
        "warmup_requests": int(warmup_requests),
        "des_p99_wait_ms": round(float(np.quantile(waits_arr, 0.99)) * 1000.0, 4),
        "des_mean_wait_ms": round(float(np.mean(waits_arr)) * 1000.0, 4),
        "des_p_wait": round(float(wait_count) / float(num_requests), 5),
    }


def run_part1_calibration(
    server_counts: list[int] | None = None,
    rho_grid: list[float] | None = None,
    num_requests: int = 100_000,
    warmup_requests: int = 15_000,
    per_vcpu_mu: float = 125.0,
    arrival_cv2: float = 1.6,
    service_cv2: float = 1.2,
    base_seed: int = 1000,
) -> dict[str, Any]:
    """Part 1: Compare Kingman-Sakasegawa analytical P99 wait vs. empirical DES P99 wait."""
    c_list = server_counts if server_counts is not None else [8, 32]
    rhos = rho_grid if rho_grid is not None else [0.55, 0.65, 0.75, 0.85, 0.90, 0.95]

    points: list[dict[str, Any]] = []
    by_server: dict[str, list[dict[str, Any]]] = {}

    seed_idx = 0
    for c in c_list:
        c_rows: list[dict[str, Any]] = []
        for rho in rhos:
            seed_idx += 1
            des_res = simulate_stationary_ggc(
                num_servers=c,
                rho=rho,
                per_vcpu_mu=per_vcpu_mu,
                arrival_cv2=arrival_cv2,
                service_cv2=service_cv2,
                num_requests=num_requests,
                warmup_requests=warmup_requests,
                seed=base_seed + seed_idx * 31,
            )
            lam = rho * float(c) * per_vcpu_mu
            ks_p99_ms = ggc_quantile_wait_seconds(
                arrival_rate_rps=lam,
                vcpus=float(c),
                per_vcpu_mu=per_vcpu_mu,
                quantile=0.99,
                arrival_cv2=arrival_cv2,
                service_cv2=service_cv2,
            ) * 1000.0
            ks_mean_ms = ggc_mean_wait_seconds(
                arrival_rate_rps=lam,
                vcpus=float(c),
                per_vcpu_mu=per_vcpu_mu,
                arrival_cv2=arrival_cv2,
                service_cv2=service_cv2,
            ) * 1000.0
            ks_p_wait = ggc_wait_probability(lam, float(c), per_vcpu_mu)

            des_p99 = float(des_res["des_p99_wait_ms"])
            signed_err = ks_p99_ms - des_p99
            abs_err = abs(signed_err)
            if signed_err >= -0.05:
                bias = "conservative"
            else:
                bias = "optimistic"

            row = {
                "c": int(c),
                "rho": float(rho),
                "arrival_rate_rps": round(lam, 3),
                "analytical_p99_ms": round(ks_p99_ms, 4),
                "des_p99_ms": round(des_p99, 4),
                "signed_error_ms": round(signed_err, 4),
                "abs_error_ms": round(abs_err, 4),
                "ks_assessment": bias,
                "analytical_mean_wait_ms": round(ks_mean_ms, 4),
                "des_mean_wait_ms": float(des_res["des_mean_wait_ms"]),
                "analytical_p_wait": round(ks_p_wait, 5),
                "des_p_wait": float(des_res["des_p_wait"]),
                "measured_requests": int(num_requests),
            }
            points.append(row)
            c_rows.append(row)
        by_server[str(c)] = c_rows

    overall_mae = float(np.mean([p["abs_error_ms"] for p in points]))
    mae_by_c = {
        str(c): round(float(np.mean([p["abs_error_ms"] for p in by_server[str(c)]])), 4)
        for c in c_list
    }
    conservative_count = sum(1 for p in points if p["ks_assessment"] == "conservative")

    return {
        "parameters": {
            "per_vcpu_mu": per_vcpu_mu,
            "mean_service_time_ms": round(1000.0 / per_vcpu_mu, 4),
            "arrival_cv2": arrival_cv2,
            "service_cv2": service_cv2,
            "requests_per_point": int(num_requests),
            "warmup_requests_per_point": int(warmup_requests),
            "server_counts": c_list,
            "rho_grid": rhos,
        },
        "summary": {
            "overall_mae_ms": round(overall_mae, 4),
            "mae_by_server_count_ms": mae_by_c,
            "conservative_points": int(conservative_count),
            "total_points": len(points),
        },
        "by_server_count": by_server,
        "points": points,
    }


def _replay_trace_with_des(
    dataset_name: str,
    csv_filename: str,
    description: str,
    max_steps: int,
    interval_sec: float,
    num_scenarios: int,
    seed: int = 404,
) -> tuple[str, dict[str, Any]]:
    """Part 2 worker: replay a real trace and evaluate 4 strategies via DES + analytical oracle."""
    timestamps, rps_trace, total_rows = load_empirical_trace(
        csv_path=DATA_DIR / csv_filename,
        scale_to_mean_rps=3200.0,
        max_steps=max_steps,
        start_offset=0,
        causal_warmup_steps=60,
    )
    policy: SLAPolicy = default_publication_policy(num_scenarios=num_scenarios)
    pools = _scale_spot_prices(default_us_east_1_pools(), spot_price_mult=1.0, interruption_rate_mult=1.0)
    pool_map = {p.pool_id: p for p in pools}
    n_steps = len(rps_trace)
    dt_hours = 5.0 / 60.0

    engine = AegisScaleEngine(pools=pools, policy=policy, seed=seed)
    copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.40)
    realized_draws = copula.sample_scenarios(pools=pools, num_scenarios=n_steps, seed=seed + 503)
    storms = _build_exogenous_storm_schedule(n_steps=n_steps, seed=seed + 19)

    od_sorted = sorted((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_od = od_sorted[0]
    spot_sorted = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    cheapest_spot = spot_sorted[0]

    # Independent DES state per strategy (carrying server-availability heap smoothly across steps)
    des_sims = {s: GGcDiscreteEventSimulator(num_servers=0) for s in DES_STRATEGIES}
    des_rng = np.random.default_rng(seed + 9001 + (0 if dataset_name == "AWS-ELB-Web" else 137))

    costs = {s: 0.0 for s in DES_STRATEGIES}
    des_breaches = {s: 0 for s in DES_STRATEGIES}
    des_storm_breaches = {s: 0 for s in DES_STRATEGIES}
    ana_breaches = {s: 0 for s in DES_STRATEGIES}
    ana_storm_breaches = {s: 0 for s in DES_STRATEGIES}
    des_p99_series: dict[str, list[float]] = {s: [] for s in DES_STRATEGIES}
    ana_p99_series: dict[str, list[float]] = {s: [] for s in DES_STRATEGIES}
    total_requests_per_strat = {s: 0 for s in DES_STRATEGIES}

    prev_rps = float(rps_trace[0])
    diurnal_period = max(12, int(round(24.0 / dt_hours)))
    seasonal_slots: dict[int, float] = {}
    seasonal_ewma = prev_rps
    seasonal_resids: list[float] = []

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
        safe_vcpu = report.required_vcpus

        # Strategy 1: Reactive HPA (55% utilization target on OD)
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

        # Strategy 2: Seasonal Predictive OD (24h diurnal slot + EWMA + 1.65*sigma guard)
        slot_next = t % diurnal_period
        slot_est = seasonal_slots.get(slot_next, seasonal_ewma)
        seasonal_pt = 0.55 * seasonal_ewma + 0.45 * slot_est
        resid_std = float(np.std(seasonal_resids[-36:], ddof=1)) if len(seasonal_resids) >= 3 else 0.15 * prev_rps
        seasonal_pred_rps = max(prev_rps, seasonal_pt + 1.65 * resid_std)
        seasonal_vcpu = required_safe_vcpus(
            seasonal_pred_rps,
            policy.per_vcpu_rps_capacity,
            policy.p99_queue_delay_sla_ms,
            0.99,
            policy.arrival_cv2,
            policy.service_cv2,
        )
        alloc_seasonal_od = {cheapest_od.pool_id: max(1, int(math.ceil(seasonal_vcpu / cheapest_od.vcpu)))}

        # Strategy 3: Greedy Cheapest Spot (10% OD / 90% cheapest Spot)
        gr_od = max(1, int(math.ceil(0.10 * safe_vcpu / cheapest_od.vcpu)))
        gr_sp = max(0, int(math.ceil(0.90 * safe_vcpu / cheapest_spot.vcpu)))
        alloc_greedy = {cheapest_od.pool_id: gr_od, cheapest_spot.pool_id: gr_sp}

        # Strategy 4: AegisScale Full
        alloc_aegis = report.portfolio.allocations

        alloc_by_strat = {
            "Reactive_HPA_55": alloc_reactive,
            "Seasonal_Predictive_OD": alloc_seasonal_od,
            "Greedy_Cheapest_Spot": alloc_greedy,
            "AegisScale_Full": alloc_aegis,
        }

        # Common Random Numbers (CRN): all 4 strategies face the exact same realized
        # stream of individual request arrivals and service demands in interval t.
        arr_times, svc_times = sample_interval_requests(
            rng=des_rng,
            arrival_rate_rps=true_rps,
            interval_sec=interval_sec,
            per_vcpu_mu=policy.per_vcpu_rps_capacity,
            arrival_cv2=policy.arrival_cv2,
            service_cv2=policy.service_cv2,
        )

        for s_name in DES_STRATEGIES:
            alloc = alloc_by_strat[s_name]
            h_cost = sum(pool_map[pid].hourly_price_usd * cnt for pid, cnt in alloc.items() if cnt > 0)
            costs[s_name] += h_cost * dt_hours

            surv_vcpu = int(
                sum(
                    pool_map[pid].vcpu * cnt
                    for pid, cnt in alloc.items()
                    if cnt > 0 and (pool_map[pid].market == "on-demand" or pid not in recalled_ids)
                )
            )

            # 1. True event-by-event DES oracle
            des_out = des_sims[s_name].run_interval(
                arrival_times=arr_times,
                service_times=svc_times,
                interval_sec=interval_sec,
                num_servers=surv_vcpu,
            )
            des_p99_ms = float(des_out["p99_wait_ms"])
            des_p99_series[s_name].append(des_p99_ms)
            total_requests_per_strat[s_name] += int(des_out["requests"])
            if des_p99_ms > policy.p99_queue_delay_sla_ms:
                des_breaches[s_name] += 1
                if storm_active:
                    des_storm_breaches[s_name] += 1

            # 2. Analytical Kingman-Sakasegawa oracle (side-by-side comparison)
            w_sec = ggc_quantile_wait_seconds(
                arrival_rate_rps=true_rps,
                vcpus=float(surv_vcpu),
                per_vcpu_mu=policy.per_vcpu_rps_capacity,
                quantile=0.99,
                arrival_cv2=policy.arrival_cv2,
                service_cv2=policy.service_cv2,
            )
            ana_p99_ms = min(HTTP_GATEWAY_TIMEOUT_MS, w_sec * 1000.0) if math.isfinite(w_sec) else HTTP_GATEWAY_TIMEOUT_MS
            ana_p99_series[s_name].append(ana_p99_ms)
            if (not math.isfinite(w_sec)) or (w_sec * 1000.0 > policy.p99_queue_delay_sla_ms):
                ana_breaches[s_name] += 1
                if storm_active:
                    ana_storm_breaches[s_name] += 1

        # Update causal seasonal state AFTER evaluating step t
        slot_cur = t % diurnal_period
        pred_used = 0.55 * seasonal_ewma + 0.45 * seasonal_slots.get(slot_cur, seasonal_ewma)
        seasonal_resids.append(true_rps - pred_used)
        seasonal_ewma = 0.35 * true_rps + 0.65 * seasonal_ewma
        if slot_cur in seasonal_slots:
            seasonal_slots[slot_cur] = 0.40 * true_rps + 0.60 * seasonal_slots[slot_cur]
        else:
            seasonal_slots[slot_cur] = true_rps
        prev_rps = true_rps

    total_hours = n_steps * dt_hours
    annual_mult = (24.0 * 365.0) / total_hours

    strat_summary: dict[str, dict[str, float | int]] = {}
    for s_name in DES_STRATEGIES:
        des_cont = (1.0 - des_breaches[s_name] / n_steps) * 100.0
        ana_cont = (1.0 - ana_breaches[s_name] / n_steps) * 100.0
        des_arr = np.asarray(des_p99_series[s_name], dtype=np.float64)
        ana_arr = np.asarray(ana_p99_series[s_name], dtype=np.float64)
        strat_summary[s_name] = {
            "des_sla_continuity_pct": round(des_cont, 2),
            "analytical_sla_continuity_pct": round(ana_cont, 2),
            "continuity_delta_des_minus_ana_pct": round(des_cont - ana_cont, 2),
            "des_total_sla_breaches": int(des_breaches[s_name]),
            "analytical_total_sla_breaches": int(ana_breaches[s_name]),
            "des_storm_sla_breaches": int(des_storm_breaches[s_name]),
            "analytical_storm_sla_breaches": int(ana_storm_breaches[s_name]),
            "des_mean_p99_wait_ms": round(float(np.mean(des_arr)), 3),
            "analytical_mean_p99_wait_ms": round(float(np.mean(ana_arr)), 3),
            "des_median_p99_wait_ms": round(float(np.median(des_arr)), 3),
            "des_max_p99_wait_ms": round(float(np.max(des_arr)), 3),
            "annualized_cost_usd": round(costs[s_name] * annual_mult, 2),
            "total_requests_simulated": int(total_requests_per_strat[s_name]),
        }

    return dataset_name, {
        "metadata": {
            "csv_file": csv_filename,
            "description": description,
            "total_csv_rows": total_rows,
            "evaluated_steps": n_steps,
            "interval_sim_seconds": interval_sec,
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "mean_rps": round(float(np.mean(rps_trace)), 2),
            "peak_rps": round(float(np.max(rps_trace)), 2),
            "total_strategy_request_events": int(sum(total_requests_per_strat.values())),
        },
        "strategies": strat_summary,
    }


def _worker_trace_task(args: tuple[str, str, str, int, float, int, int]) -> tuple[str, dict[str, Any]]:
    return _replay_trace_with_des(*args)


def generate_des_validation_figure(
    results: dict[str, Any],
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> str:
    """Render a crisp 2-panel 160-DPI publication figure for DES validation."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.4,
        }
    )

    fig, (ax_cal, ax_trace) = plt.subplots(1, 2, figsize=(14.8, 5.8))
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.98, wspace=0.20)

    # -------------------------------------------------------------------------
    # Left Panel: Kingman-Sakasegawa Analytical vs. Empirical DES P99 across rho
    # -------------------------------------------------------------------------
    part1 = results["part1_formula_calibration"]
    by_c = part1["by_server_count"]

    c_styles = {
        "8": {"color": "#2563eb", "label_prefix": "c = 8 vCPUs"},
        "32": {"color": "#10b981", "label_prefix": "c = 32 vCPUs"},
    }
    for c_key in ["8", "32"]:
        rows = by_c[c_key]
        rhos = [float(r["rho"]) for r in rows]
        ana_vals = [float(r["analytical_p99_ms"]) for r in rows]
        des_vals = [float(r["des_p99_ms"]) for r in rows]
        color = c_styles[c_key]["color"]
        prefix = c_styles[c_key]["label_prefix"]

        ax_cal.plot(
            rhos,
            ana_vals,
            linestyle="--",
            marker="^",
            linewidth=2.0,
            markersize=6.8,
            color=color,
            alpha=0.9,
            label=f"{prefix}: Kingman-Sakasegawa",
        )
        ax_cal.plot(
            rhos,
            des_vals,
            linestyle="-",
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=f"{prefix}: Empirical DES (100k req/pt)",
        )

    ax_cal.axhline(
        25.0,
        color="#dc2626",
        linestyle=":",
        linewidth=1.6,
        label="25 ms P99 SLA Threshold",
    )
    ax_cal.set_xlabel("Server Utilization (rho)")
    ax_cal.set_ylabel("P99 Queueing Delay (ms)")
    ax_cal.set_title("(a) Analytical Kingman-Sakasegawa vs. Exact G/G/c DES")
    ax_cal.set_xticks([0.55, 0.65, 0.75, 0.85, 0.90, 0.95])
    ax_cal.grid(True, linestyle="--", alpha=0.35)
    ax_cal.legend(loc="upper left", framealpha=0.94)

    mae_all = part1["summary"]["overall_mae_ms"]
    mae_8 = part1["summary"]["mae_by_server_count_ms"]["8"]
    mae_32 = part1["summary"]["mae_by_server_count_ms"]["32"]
    ax_cal.text(
        0.03,
        0.60,
        f"MAE (c=8): {mae_8:.2f} ms\nMAE (c=32): {mae_32:.2f} ms\nOverall MAE: {mae_all:.2f} ms",
        transform=ax_cal.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    # -------------------------------------------------------------------------
    # Right Panel: DES Empirical P99 SLA Continuity across 4 strategies & 2 traces
    # -------------------------------------------------------------------------
    part2 = results["part2_trace_replay"]
    ds_names = [spec[0] for spec in TRACE_SPECS]
    strat_labels = {
        "Reactive_HPA_55": "Reactive HPA (55%)",
        "Seasonal_Predictive_OD": "Seasonal Predictive OD",
        "Greedy_Cheapest_Spot": "Greedy Spot (Karpenter)",
        "AegisScale_Full": "AegisScale (Full)",
    }
    strat_colors = {
        "Reactive_HPA_55": "#64748b",
        "Seasonal_Predictive_OD": "#8b5cf6",
        "Greedy_Cheapest_Spot": "#ef4444",
        "AegisScale_Full": "#10b981",
    }

    x = np.arange(len(ds_names), dtype=np.float64)
    n_s = len(DES_STRATEGIES)
    width = 0.18
    offsets = (np.arange(n_s) - (n_s - 1) / 2.0) * width

    min_val = 100.0
    for idx, s_name in enumerate(DES_STRATEGIES):
        des_vals = [float(part2[ds]["strategies"][s_name]["des_sla_continuity_pct"]) for ds in ds_names]
        ana_vals = [float(part2[ds]["strategies"][s_name]["analytical_sla_continuity_pct"]) for ds in ds_names]
        min_val = min(min_val, min(des_vals), min(ana_vals))
        edge_w = 1.2 if s_name == "AegisScale_Full" else 0.5
        bars = ax_trace.bar(
            x + offsets[idx],
            des_vals,
            width=width,
            label=strat_labels[s_name],
            color=strat_colors[s_name],
            edgecolor="#0f172a",
            linewidth=edge_w,
        )
        # Overlay analytical continuity as crisp diamond markers for direct side-by-side comparison
        ax_trace.scatter(
            x + offsets[idx],
            ana_vals,
            marker="D",
            s=36,
            facecolor="#ffffff",
            edgecolor="#0f172a",
            linewidth=1.1,
            zorder=5,
            label="Analytical G/G/c Oracle" if idx == 0 else None,
        )
        for bar, dv, av in zip(bars, des_vals, ana_vals):
            label_y = max(dv, av)
            ax_trace.annotate(
                f"{dv:.1f}%",
                (bar.get_x() + bar.get_width() / 2.0, label_y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold" if s_name == "AegisScale_Full" else "normal",
                color="#065f46" if s_name == "AegisScale_Full" else "#1e293b",
            )

    ax_trace.axhline(98.5, color="#0f172a", linestyle="--", linewidth=1.2, alpha=0.8, label="98.5% SLA Target")
    ax_trace.set_xticks(x)
    ax_trace.set_xticklabels(ds_names, fontsize=10.2)
    ax_trace.set_ylim(max(0.0, math.floor(min_val - 8.0)), 105.5)
    ax_trace.set_ylabel("P99 Latency SLA Continuity (% of Intervals <= 25 ms)")
    ax_trace.set_title("(b) End-to-End Trace Replay: Empirical DES vs. Analytical Continuity")
    ax_trace.grid(axis="y", linestyle="--", alpha=0.35)
    ax_trace.legend(loc="lower left", ncol=2, framealpha=0.94)

    fig.suptitle(
        "Independent Event-by-Event G/G/c Discrete-Event Simulation (DES) Validation",
        y=0.95,
        fontsize=12.6,
        fontweight="bold",
    )
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return str(figure_path)


def run_discrete_event_validation(
    fast: bool = False,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> dict[str, Any]:
    """Execute Part 1 (Formula Calibration) and Part 2 (Trace Replay DES Oracle)."""
    t0 = time.perf_counter()

    part1_requests = 12_000 if fast else 100_000
    part1_warmup = 2_000 if fast else 15_000
    max_steps = 40 if fast else 200
    interval_sec = 1.0 if fast else 5.0
    num_scenarios = 80 if fast else 150

    part1 = run_part1_calibration(
        server_counts=[8, 32],
        rho_grid=[0.55, 0.65, 0.75, 0.85, 0.90, 0.95],
        num_requests=part1_requests,
        warmup_requests=part1_warmup,
        per_vcpu_mu=125.0,
        arrival_cv2=1.6,
        service_cv2=1.2,
    )

    trace_tasks = [
        (ds_name, csv_file, desc, max_steps, interval_sec, num_scenarios, 404)
        for ds_name, csv_file, desc in TRACE_SPECS
    ]

    part2: dict[str, Any] = {}
    if fast:
        for task in trace_tasks:
            ds_name, payload = _replay_trace_with_des(*task)
            part2[ds_name] = payload
    else:
        max_workers = min(len(trace_tasks), os.cpu_count() or 2)
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            futures = [executor.submit(_worker_trace_task, task) for task in trace_tasks]
            for fut in futures:
                ds_name, payload = fut.result()
                part2[ds_name] = payload

    total_des_events_part1 = len(part1["points"]) * (part1_requests + part1_warmup)
    total_des_events_part2 = sum(int(part2[ds]["metadata"]["total_strategy_request_events"]) for ds in part2)

    elapsed_sec = time.perf_counter() - t0
    results: dict[str, Any] = {
        "metadata": {
            "mode": "fast" if fast else "full",
            "elapsed_wall_time_sec": round(elapsed_sec, 2),
            "part1_total_request_events": int(total_des_events_part1),
            "part2_total_request_events": int(total_des_events_part2),
            "grand_total_request_events": int(total_des_events_part1 + total_des_events_part2),
            "sla_p99_threshold_ms": 25.0,
            "http_gateway_timeout_ms": HTTP_GATEWAY_TIMEOUT_MS,
        },
        "part1_formula_calibration": part1,
        "part2_trace_replay": part2,
    }

    fig_saved = generate_des_validation_figure(results=results, figure_path=figure_path)
    results["metadata"]["figure_path"] = fig_saved

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run independent multi-server G/G/c Discrete-Event Simulation (DES) validation."
    )
    parser.add_argument("--fast", action="store_true", help="Run fast configuration for quick verification.")
    args = parser.parse_args()

    res = run_discrete_event_validation(fast=args.fast)
    meta = res["metadata"]
    p1_sum = res["part1_formula_calibration"]["summary"]
    print(
        f"DES Validation completed in {meta['elapsed_wall_time_sec']}s ({meta['mode']}), "
        f"total simulated request events = {meta['grand_total_request_events']:,}."
    )
    print(
        f"Part 1 MAE: overall={p1_sum['overall_mae_ms']:.3f} ms "
        f"(c=8: {p1_sum['mae_by_server_count_ms']['8']:.3f} ms, "
        f"c=32: {p1_sum['mae_by_server_count_ms']['32']:.3f} ms), "
        f"conservative={p1_sum['conservative_points']}/{p1_sum['total_points']}."
    )
    for ds_name, ds_payload in res["part2_trace_replay"].items():
        print(f"\n[{ds_name}]")
        for s_name in DES_STRATEGIES:
            st = ds_payload["strategies"][s_name]
            print(
                f"  {s_name:24s} | DES SLA={st['des_sla_continuity_pct']:6.2f}% "
                f"| Ana SLA={st['analytical_sla_continuity_pct']:6.2f}% "
                f"| DES mean P99={st['des_mean_p99_wait_ms']:7.2f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

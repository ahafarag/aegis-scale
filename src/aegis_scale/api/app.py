"""Production FastAPI service for AegisScale Cloud FinOps and Continuity Control Plane."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.math_core.cvar_milp import InfeasibleCapacityError
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus

_DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


class OptimizeRequest(BaseModel):
    observed_rps: float = Field(..., ge=0.0, le=5_000_000.0, description="Current incoming requests/sec")
    policy_override: SLAPolicy | None = None


class PreemptionEventRequest(BaseModel):
    instance_type: str = Field(..., examples=["c6a.xlarge"])
    az: str = Field(..., examples=["us-east-1a"])
    observed_rps: float = Field(default=1800.0, ge=0.0)
    event_kind: str = Field(default="interruption", pattern="^(interruption|rebalance)$")


class ROISimulationRequest(BaseModel):
    base_rps: float = Field(default=2400.0, ge=100.0, le=500_000.0)
    peak_multiplier: float = Field(default=2.6, ge=1.1, le=10.0)
    steps: int = Field(default=96, ge=24, le=500, description="Number of 15-minute intervals")
    spot_storm_intensity: float = Field(default=0.75, ge=0.0, le=1.0)
    on_demand_floor_ratio: float = Field(default=0.25, ge=0.0, le=0.8)


def _run_quick_roi_comparison(req: ROISimulationRequest, pools: list[ComputePool]) -> dict[str, Any]:
    """Compare AegisScale against 100% On-Demand and Greedy Spot over a bursty trace."""
    rng = np.random.default_rng(101)
    policy = SLAPolicy(
        on_demand_floor_ratio=req.on_demand_floor_ratio,
        num_scenarios=250,
    )
    engine = AegisScaleEngine(pools=pools, policy=policy, seed=101)

    cheapest_od = min((p for p in pools if p.market == "on-demand"), key=lambda p: p.hourly_price_usd / p.vcpu)
    spot_sorted = sorted((p for p in pools if p.market == "spot"), key=lambda p: p.hourly_price_usd / p.vcpu)
    greedy_spot = spot_sorted[0]

    aegis_cost = 0.0
    od_cost = 0.0
    greedy_cost = 0.0

    aegis_sla_breaches = 0
    od_sla_breaches = 0
    greedy_sla_breaches = 0

    dt_hours = 0.25  # 15-minute intervals
    timeline: list[dict[str, float]] = []
    prev_rps = req.base_rps

    for t in range(req.steps):
        diurnal = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / 48.0 - np.pi / 2.0))
        spike = req.base_rps * 0.9 if (t in (28, 29, 68, 69)) else 0.0
        noise = float(rng.lognormal(mean=0.0, sigma=0.08))
        true_rps = float((req.base_rps * (1.0 + (req.peak_multiplier - 1.0) * diurnal) + spike) * noise)

        # Correlated Spot recall storm in AZ us-east-1a and family c6a during intervals 30..34 and 70..73
        storm_active = t in (30, 31, 32, 33, 70, 71, 72) and (
            rng.random() < req.spot_storm_intensity
        )
        compromised_az = "us-east-1a" if storm_active else None
        compromised_fam = "c6a" if storm_active else None

        if storm_active:
            engine.flag_pool_preemption_warning(greedy_spot.instance_type, greedy_spot.az, elevated_prob=0.95)
        else:
            engine.clear_pool_warnings()

        report = engine.run_cycle(observed_rps=prev_rps)
        # Re-update ACI with realized true_rps is handled on next step via prev_rps; evaluate realized true_rps on provisioned fleet:
        # Surviving vCPUs for AegisScale during storm:
        aegis_surv_vcpu = 0
        pool_map = {p.pool_id: p for p in pools}
        for pid, count in report.portfolio.allocations.items():
            p = pool_map[pid]
            recalled = (
                storm_active
                and p.market == "spot"
                and (p.az == compromised_az and p.family == compromised_fam)
            )
            if not recalled:
                aegis_surv_vcpu += p.vcpu * count

        aegis_wait_ms = ggc_quantile_wait_seconds(
            true_rps, float(aegis_surv_vcpu), policy.per_vcpu_rps_capacity, 0.99, policy.arrival_cv2, policy.service_cv2
        ) * 1000.0
        if aegis_wait_ms > policy.p99_queue_delay_sla_ms:
            aegis_sla_breaches += 1
        aegis_cost += report.portfolio.hourly_cost_usd * dt_hours

        # Baseline 1: Standard AWS Target-Tracking 100% On-Demand HPA (55% target utilization)
        od_req_vcpu = max(
            required_safe_vcpus(
                prev_rps, policy.per_vcpu_rps_capacity, policy.p99_queue_delay_sla_ms, 0.99, policy.arrival_cv2, policy.service_cv2
            ),
            prev_rps / (0.55 * policy.per_vcpu_rps_capacity),
        )
        od_units = max(1, int(np.ceil(od_req_vcpu / cheapest_od.vcpu)))
        od_vcpu = od_units * cheapest_od.vcpu
        od_wait_ms = ggc_quantile_wait_seconds(
            true_rps, float(od_vcpu), policy.per_vcpu_rps_capacity, 0.99, policy.arrival_cv2, policy.service_cv2
        ) * 1000.0
        if od_wait_ms > policy.p99_queue_delay_sla_ms:
            od_sla_breaches += 1
        od_cost += (od_units * cheapest_od.hourly_price_usd) * dt_hours

        # Baseline 2: Greedy Cheapest Spot (puts 90% in cheapest pool c6a.xlarge us-east-1a, 10% OD)
        gr_od_units = max(1, int(np.ceil(0.10 * od_req_vcpu / cheapest_od.vcpu)))
        gr_spot_units = max(0, int(np.ceil(0.95 * od_req_vcpu / greedy_spot.vcpu)))
        gr_surv_vcpu = gr_od_units * cheapest_od.vcpu + (
            0 if storm_active else gr_spot_units * greedy_spot.vcpu
        )
        gr_wait_ms = ggc_quantile_wait_seconds(
            true_rps, float(gr_surv_vcpu), policy.per_vcpu_rps_capacity, 0.99, policy.arrival_cv2, policy.service_cv2
        ) * 1000.0
        if gr_wait_ms > policy.p99_queue_delay_sla_ms:
            greedy_sla_breaches += 1
        greedy_cost += (
            gr_od_units * cheapest_od.hourly_price_usd + gr_spot_units * greedy_spot.hourly_price_usd
        ) * dt_hours

        timeline.append(
            {
                "step": float(t),
                "true_rps": round(true_rps, 1),
                "aegis_upper_rps": round(report.conformal.upper_bound, 1),
                "aegis_hourly_usd": round(report.portfolio.hourly_cost_usd, 4),
                "od_hourly_usd": round(od_units * cheapest_od.hourly_price_usd, 4),
                "storm": 1.0 if storm_active else 0.0,
            }
        )
        prev_rps = true_rps

    sim_hours = req.steps * dt_hours
    annual_scale = (24.0 * 365.0) / sim_hours
    savings_pct = max(0.0, (od_cost - aegis_cost) / od_cost * 100.0) if od_cost > 0 else 0.0

    return {
        "simulated_hours": sim_hours,
        "summary": {
            "on_demand_reactive": {
                "period_cost_usd": round(od_cost, 2),
                "annualized_cost_usd": round(od_cost * annual_scale, 2),
                "sla_continuity_pct": round((1.0 - od_sla_breaches / req.steps) * 100.0, 2),
                "sla_breaches": od_sla_breaches,
            },
            "greedy_spot": {
                "period_cost_usd": round(greedy_cost, 2),
                "annualized_cost_usd": round(greedy_cost * annual_scale, 2),
                "sla_continuity_pct": round((1.0 - greedy_sla_breaches / req.steps) * 100.0, 2),
                "sla_breaches": greedy_sla_breaches,
            },
            "aegis_scale": {
                "period_cost_usd": round(aegis_cost, 2),
                "annualized_cost_usd": round(aegis_cost * annual_scale, 2),
                "annualized_savings_vs_od_usd": round((od_cost - aegis_cost) * annual_scale, 2),
                "savings_vs_od_pct": round(savings_pct, 2),
                "sla_continuity_pct": round((1.0 - aegis_sla_breaches / req.steps) * 100.0, 2),
                "sla_breaches": aegis_sla_breaches,
            },
        },
        "timeline": timeline,
    }


def create_app() -> FastAPI:
    application = FastAPI(
        title="AegisScale Cloud FinOps & Continuity Control Plane",
        version="0.1.0",
        description=(
            "Distributionally-robust cloud capacity and EC2 Spot portfolio optimizer "
            "combining Adaptive Conformal Inference, G/G/c queuing bounds, and CVaR MILP."
        ),
    )

    pools = default_us_east_1_pools()
    state_engine = AegisScaleEngine(pools=pools, policy=SLAPolicy())

    @application.get("/", response_class=HTMLResponse)
    def serve_dashboard() -> str:
        if _DASHBOARD_PATH.exists():
            return _DASHBOARD_PATH.read_text(encoding="utf-8")
        return "<h1>AegisScale Control Plane Running</h1>"

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "version": "0.1.0",
            "candidate_pools": len(pools),
            "aci_steps_processed": state_engine.predictor.total_steps,
        }

    @application.get("/pools")
    def list_pools() -> list[dict[str, Any]]:
        return [p.model_dump() for p in pools]

    @application.post("/optimize")
    def optimize_step(req: OptimizeRequest) -> dict[str, Any]:
        try:
            active_engine = (
                AegisScaleEngine(pools=pools, policy=req.policy_override)
                if req.policy_override is not None
                else state_engine
            )
            report = active_engine.run_cycle(observed_rps=req.observed_rps)
        except InfeasibleCapacityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "observed_rps": req.observed_rps,
            "conformal_upper_rps": round(report.conformal.upper_bound, 3),
            "alpha_t": round(report.conformal.alpha_t, 5),
            "required_safe_vcpus": round(report.required_vcpus, 3),
            "expected_p99_queue_wait_ms": round(report.expected_p99_queue_wait_ms, 3),
            "provisioned_vcpus": report.portfolio.total_provisioned_vcpus,
            "on_demand_vcpus": report.portfolio.on_demand_vcpus,
            "spot_vcpus": report.portfolio.spot_vcpus,
            "active_spot_pools": report.portfolio.active_spot_pools,
            "hourly_cost_usd": round(report.portfolio.hourly_cost_usd, 5),
            "baseline_on_demand_hourly_usd": round(report.portfolio.all_on_demand_baseline_cost_usd, 5),
            "estimated_savings_pct": round(report.portfolio.estimated_savings_pct, 2),
            "cvar_shortfall_vcpu": round(report.portfolio.cvar_shortfall_vcpu, 5),
            "solve_time_ms": round(report.portfolio.solve_time_ms, 3),
            "allocations": report.portfolio.allocations,
        }

    @application.post("/preemption-warning")
    def preemption_warning(req: PreemptionEventRequest) -> dict[str, Any]:
        elevated = 0.99 if req.event_kind == "interruption" else 0.85
        flagged = state_engine.flag_pool_preemption_warning(
            instance_type=req.instance_type,
            az=req.az,
            elevated_prob=elevated,
        )
        report = state_engine.run_cycle(observed_rps=req.observed_rps)
        return {
            "flagged_pools": flagged,
            "compromised_pools_active": report.compromised_pools,
            "rebalanced_allocations": report.portfolio.allocations,
            "hourly_cost_usd": round(report.portfolio.hourly_cost_usd, 5),
            "cvar_shortfall_vcpu": round(report.portfolio.cvar_shortfall_vcpu, 5),
            "solve_time_ms": round(report.portfolio.solve_time_ms, 3),
        }

    @application.post("/simulate-roi")
    def simulate_roi(req: ROISimulationRequest) -> dict[str, Any]:
        return _run_quick_roi_comparison(req, pools)

    return application


app = create_app()

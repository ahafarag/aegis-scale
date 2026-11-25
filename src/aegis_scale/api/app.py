"""Production FastAPI service for AegisScale Cloud FinOps and Continuity Control Plane.

Hardened for public internet exposure (e.g., https://aegis.zenithcodigo.com):
  1. Payload size guard (rejects bodies > 16 KiB before JSON parsing)
  2. Strict HTTP security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options)
  3. Algorithmic complexity bounds in public-demo mode (caps MILP scenarios & trace steps)
  4. Stateless per-request isolation in public-demo mode (prevents cross-visitor state pollution)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.cloud.multi_region import SUPPORTED_REGIONS, get_region_pools
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.k8s import K8sAegisOperator, K8sClusterOverhead, PodResources
from aegis_scale.math_core.cvar_milp import InfeasibleCapacityError
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus

_DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")
_MAX_BODY_BYTES = 16 * 1024  # 16 KiB max request payload


class OptimizeRequest(BaseModel):
    observed_rps: float = Field(..., ge=0.0, le=200_000.0, description="Current incoming requests/sec")
    policy_override: SLAPolicy | None = None


class PreemptionEventRequest(BaseModel):
    instance_type: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-z0-9]+\.[a-z0-9]+$")
    az: str = Field(..., min_length=5, max_length=24, pattern=r"^[a-z]{2}-[a-z]+-\d[a-z]$")
    observed_rps: float = Field(default=1800.0, ge=0.0, le=200_000.0)
    event_kind: str = Field(default="interruption", pattern="^(interruption|rebalance)$")


class ROISimulationRequest(BaseModel):
    base_rps: float = Field(default=2400.0, ge=100.0, le=100_000.0)
    peak_multiplier: float = Field(default=2.6, ge=1.1, le=8.0)
    steps: int = Field(default=96, ge=24, le=192, description="Number of 15-minute intervals")
    spot_storm_intensity: float = Field(default=0.75, ge=0.0, le=1.0)
    on_demand_floor_ratio: float = Field(default=0.25, ge=0.0, le=0.8)


class K8sReconcileRequest(BaseModel):
    region: str = Field(default="us-east-1", max_length=24, description="Target AWS region")
    observed_rps: float = Field(default=3200.0, ge=0.0, le=200_000.0)
    pod: PodResources = Field(
        default_factory=lambda: PodResources(
            name="checkout-api",
            cpu_vcpu=1.0,
            memory_gib=3.5,
            target_rps_per_pod=120.0,
        )
    )
    overhead: K8sClusterOverhead = Field(default_factory=K8sClusterOverhead)


def _clamp_demo_policy(policy: SLAPolicy, public_demo: bool) -> SLAPolicy:
    """Prevent algorithmic complexity DoS on public endpoints by bounding MILP scenario count."""
    if public_demo and policy.num_scenarios > 250:
        return policy.model_copy(update={"num_scenarios": 250})
    return policy


def _run_quick_roi_comparison(
    req: ROISimulationRequest,
    pools: list[ComputePool],
    public_demo: bool = False,
) -> dict[str, Any]:
    """Compare AegisScale against 100% On-Demand and Greedy Spot over a bursty trace."""
    rng = np.random.default_rng(101)
    steps = min(req.steps, 120) if public_demo else req.steps
    policy = SLAPolicy(
        on_demand_floor_ratio=req.on_demand_floor_ratio,
        num_scenarios=200 if public_demo else 250,
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

    dt_hours = 0.25
    timeline: list[dict[str, float]] = []
    prev_rps = req.base_rps

    for t in range(steps):
        diurnal = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / 48.0 - np.pi / 2.0))
        spike = req.base_rps * 0.9 if (t in (28, 29, 68, 69)) else 0.0
        noise = float(rng.lognormal(mean=0.0, sigma=0.08))
        true_rps = float((req.base_rps * (1.0 + (req.peak_multiplier - 1.0) * diurnal) + spike) * noise)

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

    sim_hours = steps * dt_hours
    annual_scale = (24.0 * 365.0) / sim_hours
    savings_pct = max(0.0, (od_cost - aegis_cost) / od_cost * 100.0) if od_cost > 0 else 0.0

    return {
        "simulated_hours": sim_hours,
        "summary": {
            "on_demand_reactive": {
                "period_cost_usd": round(od_cost, 2),
                "annualized_cost_usd": round(od_cost * annual_scale, 2),
                "sla_continuity_pct": round((1.0 - od_sla_breaches / steps) * 100.0, 2),
                "sla_breaches": od_sla_breaches,
            },
            "greedy_spot": {
                "period_cost_usd": round(greedy_cost, 2),
                "annualized_cost_usd": round(greedy_cost * annual_scale, 2),
                "sla_continuity_pct": round((1.0 - greedy_sla_breaches / steps) * 100.0, 2),
                "sla_breaches": greedy_sla_breaches,
            },
            "aegis_scale": {
                "period_cost_usd": round(aegis_cost, 2),
                "annualized_cost_usd": round(aegis_cost * annual_scale, 2),
                "annualized_savings_vs_od_usd": round((od_cost - aegis_cost) * annual_scale, 2),
                "savings_vs_od_pct": round(savings_pct, 2),
                "sla_continuity_pct": round((1.0 - aegis_sla_breaches / steps) * 100.0, 2),
                "sla_breaches": aegis_sla_breaches,
            },
        },
        "timeline": timeline,
    }


def create_app(public_demo: bool | None = None) -> FastAPI:
    is_public_demo = (
        public_demo
        if public_demo is not None
        else os.environ.get("AEGIS_PUBLIC_DEMO", "false").lower() == "true"
    )

    application = FastAPI(
        title="AegisScale Cloud FinOps & Continuity Control Plane",
        version="0.2.0",
        docs_url=None if is_public_demo else "/docs",
        redoc_url=None if is_public_demo else "/redoc",
        description=(
            "Distributionally-robust cloud capacity and EC2/Karpenter Spot portfolio optimizer "
            "combining Adaptive Conformal Inference, G/G/c queuing bounds, and CVaR MILP."
        ),
    )

    @application.middleware("http")
    async def security_hardening_middleware(request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request payload too large"})

        body = await request.body()
        if len(body) > _MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request payload too large"})

        response: Response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response

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
            "version": "0.2.0",
            "public_demo_mode": is_public_demo,
            "candidate_pools": len(pools),
            "aci_steps_processed": state_engine.predictor.total_steps,
        }

    @application.get("/pools")
    def list_pools() -> list[dict[str, Any]]:
        return [p.model_dump() for p in pools]

    @application.post("/optimize")
    def optimize_step(req: OptimizeRequest) -> dict[str, Any]:
        try:
            pol = _clamp_demo_policy(req.policy_override or SLAPolicy(), is_public_demo)
            active_engine = (
                AegisScaleEngine(pools=pools, policy=pol)
                if (is_public_demo or req.policy_override is not None)
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
        # Isolate per-request in public demo mode so anonymous visitors cannot poison shared state
        target_engine = (
            AegisScaleEngine(pools=pools, policy=_clamp_demo_policy(SLAPolicy(), True))
            if is_public_demo
            else state_engine
        )
        flagged = target_engine.flag_pool_preemption_warning(
            instance_type=req.instance_type,
            az=req.az,
            elevated_prob=elevated,
        )
        report = target_engine.run_cycle(observed_rps=req.observed_rps)
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
        return _run_quick_roi_comparison(req, pools, public_demo=is_public_demo)

    @application.get("/regions")
    def list_regions() -> dict[str, Any]:
        return {
            "supported_regions": list(SUPPORTED_REGIONS),
            "pool_counts": {reg: len(get_region_pools(reg)) for reg in SUPPORTED_REGIONS},
        }

    @application.post("/k8s/reconcile")
    def reconcile_karpenter(req: K8sReconcileRequest) -> dict[str, Any]:
        if req.region not in SUPPORTED_REGIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported region '{req.region}'. Supported: {SUPPORTED_REGIONS}",
            )
        try:
            reg_pools = get_region_pools(req.region)
            pol = _clamp_demo_policy(SLAPolicy(), is_public_demo)
            k8s_op = K8sAegisOperator(
                pools=reg_pools,
                pod=req.pod,
                policy=pol,
                overhead=req.overhead,
            )
            rep = k8s_op.run_cycle(observed_rps=req.observed_rps)
        except InfeasibleCapacityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "region": req.region,
            "observed_rps": rep.observed_rps,
            "design_rps": round(rep.design_rps, 2),
            "required_pod_replicas": rep.required_pod_replicas,
            "total_schedulable_pods": rep.total_schedulable_pods,
            "expected_p99_queue_wait_ms": round(rep.expected_p99_queue_wait_ms, 3),
            "hourly_cost_usd": round(rep.portfolio.hourly_cost_usd, 5),
            "estimated_savings_pct": round(rep.portfolio.estimated_savings_pct, 2),
            "solve_time_ms": round(rep.portfolio.solve_time_ms, 3),
            "allocations": rep.portfolio.allocations,
            "manifests": {
                "ec2_node_class": rep.manifests.ec2_node_class,
                "node_pools": rep.manifests.node_pools,
                "scale_patch": rep.manifests.scale_patch,
            },
        }

    return application


app = create_app()

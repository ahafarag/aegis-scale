"""Typed configuration and policy schemas for AegisScale."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ComputePool(BaseModel):
    """Represents a single AWS EC2 provisioning pool (On-Demand or Spot in a given AZ)."""

    pool_id: str = Field(..., description="Unique pool identifier, e.g., spot-m7g.xlarge-az1")
    instance_type: str = Field(..., description="EC2 instance type, e.g., m7g.xlarge")
    family: str = Field(..., description="Instance family for correlation grouping, e.g., m7g")
    az: str = Field(..., description="Availability Zone identifier, e.g., us-east-1a")
    market: Literal["on-demand", "spot"] = Field(..., description="Pricing market")
    vcpu: int = Field(..., ge=1, description="Effective vCPUs per instance")
    memory_gib: float = Field(..., gt=0.0, description="Memory in GiB per instance")
    hourly_price_usd: float = Field(..., gt=0.0, description="Current hourly cost in USD")
    base_interruption_prob: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Marginal per-interval Spot interruption probability (0 for on-demand)",
    )
    max_instances: int = Field(default=200, ge=1, description="Per-pool quota or scaling limit")

    @model_validator(mode="after")
    def _validate_on_demand_hazard(self) -> "ComputePool":
        if self.market == "on-demand" and self.base_interruption_prob > 0.0:
            raise ValueError("On-demand pools must have base_interruption_prob == 0.0")
        return self


class SLAPolicy(BaseModel):
    """Mathematical SLA continuity and risk-control policy parameters."""

    alpha_target: float = Field(
        default=0.02,
        gt=0.0,
        lt=0.5,
        description="Target miscoverage rate for Adaptive Conformal Inference (e.g., 0.02 = 98% coverage)",
    )
    aci_gamma: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Step size gamma for online ACI quantile adaptation",
    )
    aci_window: int = Field(
        default=120,
        ge=20,
        description="Sliding calibration window size for conformal residuals",
    )
    per_vcpu_rps_capacity: float = Field(
        default=125.0,
        gt=0.0,
        description="Mean service rate mu (requests/sec per effective vCPU)",
    )
    arrival_cv2: float = Field(
        default=1.6,
        ge=0.0,
        description="Squared coefficient of variation of inter-arrival times c_a^2",
    )
    service_cv2: float = Field(
        default=1.2,
        ge=0.0,
        description="Squared coefficient of variation of service times c_s^2",
    )
    p99_queue_delay_sla_ms: float = Field(
        default=25.0,
        gt=0.0,
        description="Maximum permissible P99 queueing wait W_{q,0.99} in milliseconds",
    )
    cvar_beta: float = Field(
        default=0.95,
        gt=0.5,
        lt=0.9999,
        description="Confidence level beta for Conditional Value-at-Risk (e.g., 0.95 or 0.99)",
    )
    max_cvar_shortfall_vcpu: float = Field(
        default=0.0,
        ge=0.0,
        description="Hard upper bound on CVaR_beta capacity shortfall in vCPUs",
    )
    max_cvar_shortfall_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Proportional CVaR_beta shortfall budget as a fraction of C_req (effective cap = max(vcpu, ratio * C_req))",
    )
    on_demand_floor_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of C_req backed by zero-risk On-Demand capacity",
    )
    max_spot_pool_share: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description="Maximum fraction of C_req placed in any single Spot pool",
    )
    churn_penalty_per_vcpu_usd: float = Field(
        default=0.004,
        ge=0.0,
        description="Switching cost per vCPU changed to prevent cold-start thrashing",
    )
    cvar_penalty_weight: float = Field(
        default=10.0,
        ge=0.0,
        description="Objective penalty weight lambda_cvar on CVaR shortfall",
    )
    num_scenarios: int = Field(
        default=400,
        ge=50,
        le=5000,
        description="Number of correlated preemption scenarios S in the MILP",
    )
    az_correlation: float = Field(
        default=0.55,
        ge=0.0,
        lt=0.99,
        description="Within-AZ latent factor correlation for Spot recalls",
    )
    family_correlation: float = Field(
        default=0.40,
        ge=0.0,
        lt=0.99,
        description="Within-instance-family latent factor correlation for Spot recalls",
    )

    @model_validator(mode="after")
    def _validate_correlations(self) -> "SLAPolicy":
        if self.az_correlation + self.family_correlation >= 0.99:
            raise ValueError("Sum of az_correlation and family_correlation must be < 0.99")
        return self

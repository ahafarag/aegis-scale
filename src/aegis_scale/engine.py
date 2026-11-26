"""End-to-end AegisScale control-loop engine uniting ACI, G/G/c queuing, and CVaR MILP."""

from __future__ import annotations

from dataclasses import dataclass
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor, ConformalStepResult
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds, required_safe_vcpus
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer, PortfolioDecision


@dataclass(frozen=True)
class ControlCycleReport:
    """Complete telemetry and optimization record for a single AegisScale control step."""

    conformal: ConformalStepResult
    required_vcpus: float
    expected_p99_queue_wait_ms: float
    portfolio: PortfolioDecision
    compromised_pools: list[str]


class AegisScaleEngine:
    """Stateful control plane orchestrating online ACI forecasting, G/G/c sizing, and CVaR MILP."""

    def __init__(
        self,
        pools: list[ComputePool],
        policy: SLAPolicy | None = None,
        seed: int = 42,
    ) -> None:
        if not pools:
            raise ValueError("pools must contain at least one ComputePool")
        self.pools = list(pools)
        self.policy = policy or SLAPolicy()
        self.seed = seed

        self.predictor = AdaptiveConformalPredictor(
            alpha_target=self.policy.alpha_target,
            gamma=self.policy.aci_gamma,
            window_size=self.policy.aci_window,
        )
        self.hazard_model = CorrelatedSpotHazardModel(
            az_correlation=self.policy.az_correlation,
            family_correlation=self.policy.family_correlation,
        )
        self.optimizer = CVaRPortfolioOptimizer(policy=self.policy)

        self.current_allocations: dict[str, int] | None = None
        self.elevated_hazards: dict[str, float] = {}
        self._cycle_index = 0

    def flag_pool_preemption_warning(
        self,
        instance_type: str,
        az: str,
        elevated_prob: float = 0.98,
    ) -> list[str]:
        """Mark matching Spot pool(s) as compromised upon AWS 2-min interruption or rebalance event."""
        flagged: list[str] = []
        for pool in self.pools:
            if pool.market == "spot" and pool.instance_type == instance_type and pool.az == az:
                self.elevated_hazards[pool.pool_id] = max(0.5, min(0.999, float(elevated_prob)))
                flagged.append(pool.pool_id)
        return flagged

    def clear_pool_warnings(self) -> None:
        """Reset transient elevated hazard flags after cool-down."""
        self.elevated_hazards.clear()

    def run_cycle(self, observed_rps: float) -> ControlCycleReport:
        """Ingest latest RPS telemetry, update ACI, size capacity via G/G/c, and solve CVaR MILP."""
        conformal_res = self.predictor.step(observed_rps)
        _, next_upper_rps = self.predictor.predict_upper_bound()
        # Next-interval conformal upper bound floored at current observed RPS
        design_rps = max(next_upper_rps, observed_rps)

        c_req = required_safe_vcpus(
            arrival_rate_rps=design_rps,
            per_vcpu_mu=self.policy.per_vcpu_rps_capacity,
            sla_wait_ms=self.policy.p99_queue_delay_sla_ms,
            quantile=0.99,
            arrival_cv2=self.policy.arrival_cv2,
            service_cv2=self.policy.service_cv2,
        )

        scenarios = self.hazard_model.sample_scenarios(
            pools=self.pools,
            num_scenarios=self.policy.num_scenarios,
            elevated_pool_hazards=self.elevated_hazards,
            seed=self.seed + self._cycle_index,
        )
        self._cycle_index += 1

        decision = self.optimizer.solve(
            pools=self.pools,
            required_vcpus=c_req,
            scenarios=scenarios,
            previous_allocations=self.current_allocations,
        )
        self.current_allocations = dict(decision.allocations)

        p99_wait_sec = ggc_quantile_wait_seconds(
            arrival_rate_rps=design_rps,
            vcpus=float(decision.total_provisioned_vcpus),
            per_vcpu_mu=self.policy.per_vcpu_rps_capacity,
            quantile=0.99,
            arrival_cv2=self.policy.arrival_cv2,
            service_cv2=self.policy.service_cv2,
        )

        return ControlCycleReport(
            conformal=conformal_res,
            required_vcpus=c_req,
            expected_p99_queue_wait_ms=p99_wait_sec * 1000.0,
            portfolio=decision,
            compromised_pools=sorted(self.elevated_hazards.keys()),
        )

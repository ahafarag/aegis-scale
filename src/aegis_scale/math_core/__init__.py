"""Mathematical core for AegisScale: ACI, G/G/c queuing bounds, hazard copula, and CVaR MILP."""

from aegis_scale.math_core.conformal import AdaptiveConformalPredictor, ConformalStepResult
from aegis_scale.math_core.queuing import (
    ggc_mean_wait_seconds,
    ggc_quantile_wait_seconds,
    required_safe_vcpus,
)
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer, PortfolioDecision, empirical_cvar

__all__ = [
    "AdaptiveConformalPredictor",
    "ConformalStepResult",
    "ggc_mean_wait_seconds",
    "ggc_quantile_wait_seconds",
    "required_safe_vcpus",
    "CorrelatedSpotHazardModel",
    "CVaRPortfolioOptimizer",
    "PortfolioDecision",
    "empirical_cvar",
]

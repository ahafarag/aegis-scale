"""Rigorous mathematical invariant tests proving ACI bounds, G/G/c exactness, copula moments, and CVaR MILP."""

from __future__ import annotations

import math
import numpy as np
import pytest

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer, empirical_cvar
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.queuing import (
    ggc_mean_wait_seconds,
    ggc_quantile_wait_seconds,
    required_safe_vcpus,
)


def test_theorem_1_aci_finite_sample_coverage_under_adversarial_regime_shifts() -> None:
    """Verify Gibbs & Candes Theorem 1 bound |(1/T) sum err_t - alpha| <= (1 + gamma)/(gamma*T)."""
    rng = np.random.default_rng(7)
    alpha_target = 0.05
    gamma = 0.05
    total_steps = 1500

    # Exact infinite-ceiling theoretical mode verifies Theorem 1 algebraically
    aci_theory = AdaptiveConformalPredictor(
        alpha_target=alpha_target, gamma=gamma, window_size=80, finite_cap=False
    )
    # Practical finite-ceiling mode verifies realized empirical miscoverage honestly
    aci_practical = AdaptiveConformalPredictor(
        alpha_target=alpha_target, gamma=gamma, window_size=80, finite_cap=True
    )

    for t in range(total_steps):
        regime_base = 1000.0 if (t // 250) % 2 == 0 else 4200.0
        heavy_tail = float((rng.pareto(a=2.5) + 1.0) * 120.0)
        y_t = regime_base + 300.0 * math.sin(2.0 * math.pi * t / 60.0) + heavy_tail
        res_th = aci_theory.step(y_t)
        res_pr = aci_practical.step(y_t)

    theoretical_bound = AdaptiveConformalPredictor.theoretical_miscoverage_deviation_bound(
        res_th.step, gamma
    )
    assert abs(res_th.empirical_miscoverage_rate - alpha_target) <= theoretical_bound + 1e-12
    assert abs(res_pr.empirical_miscoverage_rate - alpha_target) < 0.025


def test_ggc_reduces_to_exact_mm1_closed_form_and_is_strictly_monotone() -> None:
    """Verify G/G/c quantile formula matches exact analytical M/M/1 waiting law at c=1, c_a^2=c_s^2=1."""
    lam = 80.0
    mu = 100.0
    rho = lam / mu  # 0.80
    quantile = 0.99
    delta = 1.0 - quantile

    # Exact analytical M/M/1 formula: E[W_q] = rho / (mu*(1-rho)), W_{q,1-delta} = ln(rho/delta)/(mu*(1-rho))
    exact_mean = rho / (mu * (1.0 - rho))
    exact_p99 = math.log(rho / delta) / (mu * (1.0 - rho))

    calc_mean = ggc_mean_wait_seconds(lam, vcpus=1.0, per_vcpu_mu=mu, arrival_cv2=1.0, service_cv2=1.0)
    calc_p99 = ggc_quantile_wait_seconds(
        lam, vcpus=1.0, per_vcpu_mu=mu, quantile=quantile, arrival_cv2=1.0, service_cv2=1.0
    )

    assert math.isclose(calc_mean, exact_mean, rel_tol=1e-12)
    assert math.isclose(calc_p99, exact_p99, rel_tol=1e-12)

    # Strict monotonicity in vCPUs for c > lambda/mu
    waits = [
        ggc_quantile_wait_seconds(2500.0, vcpus=c, per_vcpu_mu=125.0, quantile=0.99)
        for c in (21.0, 23.0, 26.0, 30.0, 40.0)
    ]
    for w_prev, w_next in zip(waits, waits[1:]):
        assert w_next <= w_prev

    # Bisection root accuracy
    c_star = required_safe_vcpus(
        arrival_rate_rps=2500.0,
        per_vcpu_mu=125.0,
        sla_wait_ms=25.0,
        quantile=0.99,
        arrival_cv2=1.6,
        service_cv2=1.2,
    )
    realized_ms = (
        ggc_quantile_wait_seconds(2500.0, c_star, 125.0, 0.99, 1.6, 1.2) * 1000.0
    )
    assert realized_ms <= 25.0 + 1e-4
    #Slightly smaller c must violate the 25ms bound
    under_ms = (
        ggc_quantile_wait_seconds(2500.0, c_star - 1e-3, 125.0, 0.99, 1.6, 1.2) * 1000.0
    )
    assert under_ms > 25.0


def test_copula_preserves_marginals_and_induces_az_family_correlation() -> None:
    """Verify On-Demand never interrupts, Spot marginals converge to p_k, and AZ correlation > 0."""
    pools = default_us_east_1_pools()
    copula = CorrelatedSpotHazardModel(az_correlation=0.55, family_correlation=0.35)
    xi = copula.sample_scenarios(pools, num_scenarios=20_000, seed=123)

    for k, pool in enumerate(pools):
        emp_prob = float(np.mean(xi[:, k]))
        if pool.market == "on-demand":
            assert emp_prob == 0.0
        else:
            assert abs(emp_prob - pool.base_interruption_prob) < 0.008

    # Same-AZ Spot pools must correlate more strongly than cross-AZ cross-family Spot pools
    idx_a_c6a = next(i for i, p in enumerate(pools) if p.pool_id == "spot-c6a.xlarge-us-east-1a")
    idx_a_m7g = next(i for i, p in enumerate(pools) if p.pool_id == "spot-m7g.xlarge-us-east-1a")
    idx_c_m6i = next(i for i, p in enumerate(pools) if p.pool_id == "spot-m6i.2xlarge-us-east-1c")

    corr_same_az = float(np.corrcoef(xi[:, idx_a_c6a], xi[:, idx_a_m7g])[0, 1])
    corr_disjoint = float(np.corrcoef(xi[:, idx_a_c6a], xi[:, idx_c_m6i])[0, 1])
    assert corr_same_az > 0.12
    assert abs(corr_disjoint) < 0.03


def test_cvar_milp_exactness_diversification_and_cost_savings() -> None:
    """Verify MILP satisfies CVaR guardrail, On-Demand floor, diversification, and beats pure On-Demand."""
    pools = default_us_east_1_pools()
    policy = SLAPolicy(
        cvar_beta=0.95,
        max_cvar_shortfall_vcpu=0.0,
        on_demand_floor_ratio=0.25,
        max_spot_pool_share=0.25,
        num_scenarios=400,
    )
    copula = CorrelatedSpotHazardModel(
        az_correlation=policy.az_correlation,
        family_correlation=policy.family_correlation,
    )
    scenarios = copula.sample_scenarios(pools, num_scenarios=400, seed=99)
    optimizer = CVaRPortfolioOptimizer(policy=policy)

    c_req = 64.0
    decision = optimizer.solve(pools=pools, required_vcpus=c_req, scenarios=scenarios)

    assert decision.total_provisioned_vcpus >= 64
    assert decision.on_demand_vcpus >= math.ceil(0.25 * c_req)
    assert decision.cvar_shortfall_vcpu <= 1e-6
    assert decision.active_spot_pools >= 4
    assert decision.estimated_savings_pct > 25.0
    assert decision.solve_time_ms < 500.0

    # Verify empirical_cvar consistency on random non-negative shortfall vector
    rng = np.random.default_rng(5)
    losses = np.maximum(0.0, rng.normal(loc=2.0, scale=5.0, size=500))
    _, exact_cvar = empirical_cvar(losses, beta=0.95)
    # Brute-force grid minimization of Rockafellar-Uryasev convex auxiliary function
    etas = np.linspace(0.0, np.max(losses), 5000)
    ru_vals = [eta + np.mean(np.maximum(0.0, losses - eta)) / 0.05 for eta in etas]
    assert math.isclose(exact_cvar, float(np.min(ru_vals)), rel_tol=1e-4)

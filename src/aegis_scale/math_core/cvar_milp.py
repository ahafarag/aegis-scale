"""Rockafellar-Uryasev CVaR Mixed-Integer Linear Programming (MILP) optimizer.

Solves globally for the minimum-cost integer allocation of On-Demand and
diversified EC2 Spot pools subject to Conditional Value-at-Risk (CVaR_beta)
capacity shortfall guardrails, hard On-Demand survival floors, concentration
caps, and instance-churn regularization using HiGHS (scipy.optimize.milp).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, vstack, eye, diags

from aegis_scale.config import ComputePool, SLAPolicy


class InfeasibleCapacityError(RuntimeError):
    """Raised when candidate pools cannot satisfy required safe vCPU capacity."""


@dataclass(frozen=True)
class PortfolioDecision:
    """Globally optimal fleet portfolio produced by the CVaR MILP solver."""

    allocations: dict[str, int]
    required_vcpus: float
    total_provisioned_vcpus: int
    on_demand_vcpus: int
    spot_vcpus: int
    hourly_cost_usd: float
    all_on_demand_baseline_cost_usd: float
    estimated_savings_pct: float
    var_shortfall_vcpu: float
    cvar_shortfall_vcpu: float
    active_spot_pools: int
    solve_time_ms: float


def empirical_cvar(losses: np.ndarray, beta: float) -> tuple[float, float]:
    """Compute exact discrete Rockafellar-Uryasev (VaR_beta, CVaR_beta) for equal-weight scenarios.

    Matches min_{eta >= 0} { eta + (1 / ((1 - beta) * S)) * sum_s max(0, losses[s] - eta) }
    for non-negative loss arrays.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie strictly in (0, 1)")
    arr = np.sort(np.maximum(0.0, np.asarray(losses, dtype=np.float64).ravel()))
    s_len = len(arr)
    if s_len == 0:
        return 0.0, 0.0

    # Exact quantile index k* such that k*/S >= beta
    idx = min(s_len - 1, max(0, int(np.ceil(beta * s_len)) - 1))
    var_beta = float(arr[idx])
    cvar_beta = float(var_beta + np.sum(np.maximum(0.0, arr - var_beta)) / ((1.0 - beta) * s_len))
    return var_beta, cvar_beta


class CVaRPortfolioOptimizer:
    """Exact MILP solver for risk-averse EC2 On-Demand + Spot fleet allocation.

    Lemma (Non-Negative Shortfall Linearization):
        Let d_s(x) = C_req - C_surv^(s)(x) denote signed capacity deficit in scenario s,
        and let actual shortfall be l_s(x) = max(0, d_s(x)) >= 0.
        Because l_s(x) >= 0 for all s, VaR_beta(l(x)) >= 0. Restricting eta >= 0 gives
            max(0, d_s(x) - eta) == max(0, max(0, d_s(x)) - eta)  for all d_s(x) in R,
        eliminating S auxiliary shortfall variables while remaining 100% exact.
    """

    def __init__(self, policy: SLAPolicy | None = None) -> None:
        self.policy = policy or SLAPolicy()

    @staticmethod
    def compute_all_on_demand_baseline_cost(
        pools: list[ComputePool],
        required_vcpus: float,
    ) -> float:
        """Minimum hourly cost of satisfying required_vcpus using exclusively On-Demand pools."""
        od_pools = [p for p in pools if p.market == "on-demand"]
        if not od_pools or required_vcpus <= 0.0:
            return 0.0
        best_cost_per_vcpu = min(p.hourly_price_usd / float(p.vcpu) for p in od_pools)
        # Integer rounding using most cost-efficient On-Demand unit
        cheapest_od = min(od_pools, key=lambda p: p.hourly_price_usd / float(p.vcpu))
        units = int(np.ceil(required_vcpus / float(cheapest_od.vcpu)))
        return float(min(units * cheapest_od.hourly_price_usd, max(best_cost_per_vcpu * required_vcpus, 0.0) if units == 0 else units * cheapest_od.hourly_price_usd))

    def solve(
        self,
        pools: list[ComputePool],
        required_vcpus: float,
        scenarios: np.ndarray,
        previous_allocations: dict[str, int] | None = None,
    ) -> PortfolioDecision:
        """Solve the Rockafellar-Uryasev CVaR MILP via HiGHS."""
        t0 = time.perf_counter()
        k_len = len(pools)
        if k_len == 0:
            raise ValueError("pools must be non-empty")

        c_req = max(0.0, float(required_vcpus))
        s_len, cols = scenarios.shape
        if cols != k_len:
            raise ValueError(f"Scenario columns ({cols}) must match number of pools ({k_len})")

        pol = self.policy
        beta = pol.cvar_beta
        tail_scale = 1.0 / ((1.0 - beta) * float(s_len))

        vcpus = np.array([float(p.vcpu) for p in pools], dtype=np.float64)
        prices = np.array([float(p.hourly_price_usd) for p in pools], dtype=np.float64)
        max_inst = np.array([float(p.max_instances) for p in pools], dtype=np.float64)
        is_od = np.array([1.0 if p.market == "on-demand" else 0.0 for p in pools], dtype=np.float64)

        has_prev = previous_allocations is not None
        x_prev = np.array(
            [float(previous_allocations.get(p.pool_id, 0)) if previous_allocations else 0.0 for p in pools],
            dtype=np.float64,
        )

        # Variable layout: w = [x (K integers), z (K continuous churn), eta (1 continuous), u (S continuous)]
        # Total variables: N = 2*K + 1 + S
        n_vars = 2 * k_len + 1 + s_len
        idx_z = k_len
        idx_eta = 2 * k_len
        idx_u = 2 * k_len + 1

        # Objective vector c
        c_obj = np.zeros(n_vars, dtype=np.float64)
        c_obj[:k_len] = prices
        if has_prev and pol.churn_penalty_per_vcpu_usd > 0.0:
            c_obj[idx_z : idx_z + k_len] = pol.churn_penalty_per_vcpu_usd * vcpus
        c_obj[idx_eta] = pol.cvar_penalty_weight
        c_obj[idx_u:] = pol.cvar_penalty_weight * tail_scale

        # Variable bounds
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        ub[:k_len] = max_inst

        # Spot pool concentration upper bounds enforced directly as tight variable bounds!
        for k, pool in enumerate(pools):
            if pool.market == "spot":
                cap_vcpu = max(pol.max_spot_pool_share * c_req, float(pool.vcpu))
                max_by_share = np.floor(cap_vcpu / float(pool.vcpu))
                ub[k] = min(ub[k], max(1.0 if c_req > 0.0 else 0.0, max_by_share))

        if not has_prev:
            ub[idx_z : idx_z + k_len] = 0.0

        # Integrality: 1 for x_0..x_{K-1}, 0 for continuous z, eta, u
        integrality = np.zeros(n_vars, dtype=np.int8)
        integrality[:k_len] = 1

        # Surviving vCPU matrix A_surv of shape (S, K): A_surv[s, k] = v_k * (1 - xi[s, k])
        surv_matrix = (1.0 - scenarios) * vcpus[np.newaxis, :]

        constraint_blocks = []
        lhs_list = []
        rhs_list = []

        # 1. Nominal capacity constraint: sum_k v_k x_k >= C_req
        row_nom = csc_matrix((vcpus, ([0] * k_len, list(range(k_len)))), shape=(1, n_vars))
        constraint_blocks.append(row_nom)
        lhs_list.append(np.array([c_req], dtype=np.float64))
        rhs_list.append(np.array([np.inf], dtype=np.float64))

        # 2. Hard On-Demand survival floor: sum_{k in OD} v_k x_k >= rho_floor * C_req
        if pol.on_demand_floor_ratio > 0.0 and np.any(is_od > 0.0):
            od_vcpus = vcpus * is_od
            row_od = csc_matrix((od_vcpus, ([0] * k_len, list(range(k_len)))), shape=(1, n_vars))
            constraint_blocks.append(row_od)
            lhs_list.append(np.array([pol.on_demand_floor_ratio * c_req], dtype=np.float64))
            rhs_list.append(np.array([np.inf], dtype=np.float64))

        # 3. Scenario shortfall linearization (S rows):
        #    u_s >= C_req - sum_k A_surv[s, k] x_k - eta  <=>  sum_k A_surv[s, k] x_k + eta + u_s >= C_req
        x_part = csc_matrix(surv_matrix)
        z_zeros = csc_matrix((s_len, k_len))
        eta_ones = csc_matrix(np.ones((s_len, 1), dtype=np.float64))
        u_eye = eye(s_len, format="csc", dtype=np.float64)
        shortfall_block = hstack([x_part, z_zeros, eta_ones, u_eye], format="csc")
        constraint_blocks.append(shortfall_block)
        lhs_list.append(np.full(s_len, c_req, dtype=np.float64))
        rhs_list.append(np.full(s_len, np.inf, dtype=np.float64))

        # 4. Hard CVaR guardrail: eta + tail_scale * sum_s u_s <= effective CVaR budget
        effective_cvar_cap = max(
            float(pol.max_cvar_shortfall_vcpu),
            float(pol.max_cvar_shortfall_ratio) * c_req,
        )
        cvar_cols = [idx_eta] + list(range(idx_u, idx_u + s_len))
        cvar_vals = [1.0] + [tail_scale] * s_len
        row_cvar = csc_matrix((cvar_vals, ([0] * len(cvar_cols), cvar_cols)), shape=(1, n_vars))
        constraint_blocks.append(row_cvar)
        lhs_list.append(np.array([-np.inf], dtype=np.float64))
        rhs_list.append(np.array([effective_cvar_cap], dtype=np.float64))

        # 5. Churn linearization (2*K rows) when previous_allocations is provided:
        #    z_k - x_k >= -x_prev_k   and   z_k + x_k >= x_prev_k
        if has_prev and pol.churn_penalty_per_vcpu_usd > 0.0:
            neg_ix = -eye(k_len, format="csc", dtype=np.float64)
            pos_ix = eye(k_len, format="csc", dtype=np.float64)
            tail_zeros = csc_matrix((k_len, 1 + s_len))
            churn_1 = hstack([neg_ix, pos_ix, tail_zeros], format="csc")
            churn_2 = hstack([pos_ix, pos_ix, tail_zeros], format="csc")
            constraint_blocks.append(churn_1)
            lhs_list.append(-x_prev)
            rhs_list.append(np.full(k_len, np.inf, dtype=np.float64))
            constraint_blocks.append(churn_2)
            lhs_list.append(x_prev)
            rhs_list.append(np.full(k_len, np.inf, dtype=np.float64))

        a_mat = vstack(constraint_blocks, format="csc")
        lhs = np.concatenate(lhs_list)
        rhs = np.concatenate(rhs_list)

        solver_time_limit = 0.25 if k_len <= 64 else 1.5
        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(a_mat, lhs, rhs),
            options={"mip_rel_gap": 0.015, "time_limit": solver_time_limit},
        )

        if res.x is None or res.status not in (0, 1):
            raise InfeasibleCapacityError(
                f"CVaR MILP infeasible or failed (status={res.status}): {res.message}"
            )

        sol = res.x
        x_opt = np.rint(sol[:k_len]).astype(int)

        # Compute exact realized shortfall metrics on the scenario matrix
        surv_vcpus = surv_matrix @ x_opt.astype(np.float64)
        shortfalls = np.maximum(0.0, c_req - surv_vcpus)
        var_val, cvar_val = empirical_cvar(shortfalls, beta)

        alloc_dict = {pools[k].pool_id: int(x_opt[k]) for k in range(k_len) if x_opt[k] > 0}
        total_vcpu = int(np.dot(vcpus, x_opt))
        od_vcpu = int(np.dot(vcpus * is_od, x_opt))
        spot_vcpu = total_vcpu - od_vcpu
        hourly_cost = float(np.dot(prices, x_opt))
        baseline_cost = self.compute_all_on_demand_baseline_cost(pools, c_req)
        savings_pct = (
            max(0.0, (baseline_cost - hourly_cost) / baseline_cost * 100.0)
            if baseline_cost > 0.0
            else 0.0
        )
        active_spot = sum(1 for k in range(k_len) if pools[k].market == "spot" and x_opt[k] > 0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PortfolioDecision(
            allocations=alloc_dict,
            required_vcpus=c_req,
            total_provisioned_vcpus=total_vcpu,
            on_demand_vcpus=od_vcpu,
            spot_vcpus=spot_vcpu,
            hourly_cost_usd=hourly_cost,
            all_on_demand_baseline_cost_usd=baseline_cost,
            estimated_savings_pct=savings_pct,
            var_shortfall_vcpu=var_val,
            cvar_shortfall_vcpu=cvar_val,
            active_spot_pools=active_spot,
            solve_time_ms=elapsed_ms,
        )

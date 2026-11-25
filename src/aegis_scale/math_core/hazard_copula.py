"""Multi-factor Gaussian copula for correlated AWS EC2 Spot preemption scenarios.

Models joint Spot recalls driven by shared Availability Zone capacity pressure
and shared instance-family recalls while preserving exact marginal hazard rates.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm

from aegis_scale.config import ComputePool


class CorrelatedSpotHazardModel:
    """Three-factor Gaussian copula scenario generator for EC2 Spot pools.

    For pool k in AZ a(k) and family f(k) with marginal interruption probability p_k:
        Z_k = sqrt(rho_az) * G_{a(k)} + sqrt(rho_fam) * H_{f(k)} + sqrt(1 - rho_az - rho_fam) * eps_k
    where G_a, H_f, eps_k ~ i.i.d. N(0, 1).
    Setting xi_k = I(Z_k < Phi^{-1}(p_k)) guarantees exact marginal probability P(xi_k = 1) = p_k
    and exact latent pairwise correlations:
        Corr(Z_j, Z_k) = rho_az * I(a(j)==a(k)) + rho_fam * I(f(j)==f(k))  for j != k.
    """

    def __init__(
        self,
        az_correlation: float = 0.55,
        family_correlation: float = 0.40,
    ) -> None:
        if az_correlation < 0.0 or family_correlation < 0.0:
            raise ValueError("Correlations must be non-negative")
        if az_correlation + family_correlation >= 0.999:
            raise ValueError("az_correlation + family_correlation must be strictly < 1")

        self.az_correlation = float(az_correlation)
        self.family_correlation = float(family_correlation)
        self.idiosyncratic_weight = math.sqrt(1.0 - self.az_correlation - self.family_correlation)
        self.az_weight = math.sqrt(self.az_correlation)
        self.fam_weight = math.sqrt(self.family_correlation)

    def latent_correlation_matrix(self, pools: list[ComputePool]) -> np.ndarray:
        """Return exact K x K latent Gaussian correlation matrix across pools."""
        k_len = len(pools)
        corr = np.eye(k_len, dtype=np.float64)
        for i in range(k_len):
            for j in range(i + 1, k_len):
                if pools[i].market == "on-demand" or pools[j].market == "on-demand":
                    c = 0.0
                else:
                    c = 0.0
                    if pools[i].az == pools[j].az:
                        c += self.az_correlation
                    if pools[i].family == pools[j].family:
                        c += self.family_correlation
                corr[i, j] = c
                corr[j, i] = c
        return corr

    def sample_scenarios(
        self,
        pools: list[ComputePool],
        num_scenarios: int = 400,
        elevated_pool_hazards: dict[str, float] | None = None,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Generate binary preemption matrix xi of shape (num_scenarios, len(pools)).

        xi[s, k] == 1 indicates pool k suffers capacity recall in scenario s.
        On-Demand pools always satisfy xi[s, k] == 0.
        """
        if num_scenarios < 1:
            raise ValueError("num_scenarios must be >= 1")
        k_len = len(pools)
        if k_len == 0:
            return np.zeros((num_scenarios, 0), dtype=np.float64)

        rng = np.random.default_rng(seed)
        elevated = elevated_pool_hazards or {}

        az_list = sorted({p.az for p in pools})
        fam_list = sorted({p.family for p in pools})
        az_idx = {az: idx for idx, az in enumerate(az_list)}
        fam_idx = {fam: idx for idx, fam in enumerate(fam_list)}

        # Antithetic-stratified normal draws when num_scenarios >= 2 for variance reduction
        half = num_scenarios // 2
        if half > 0:
            g_half = rng.standard_normal((half, len(az_list)))
            h_half = rng.standard_normal((half, len(fam_list)))
            e_half = rng.standard_normal((half, k_len))
            g_draws = np.vstack([g_half, -g_half])
            h_draws = np.vstack([h_half, -h_half])
            e_draws = np.vstack([e_half, -e_half])
            if num_scenarios % 2 == 1:
                g_draws = np.vstack([g_draws, rng.standard_normal((1, len(az_list)))])
                h_draws = np.vstack([h_draws, rng.standard_normal((1, len(fam_list)))])
                e_draws = np.vstack([e_draws, rng.standard_normal((1, k_len))])
        else:
            g_draws = rng.standard_normal((num_scenarios, len(az_list)))
            h_draws = rng.standard_normal((num_scenarios, len(fam_list)))
            e_draws = rng.standard_normal((num_scenarios, k_len))

        xi = np.zeros((num_scenarios, k_len), dtype=np.float64)
        for k, pool in enumerate(pools):
            if pool.market == "on-demand":
                continue
            p_k = float(elevated.get(pool.pool_id, pool.base_interruption_prob))
            p_k = max(0.0, min(0.999999, p_k))
            if p_k <= 0.0:
                continue
            threshold = float(norm.ppf(p_k))
            z_k = (
                self.az_weight * g_draws[:, az_idx[pool.az]]
                + self.fam_weight * h_draws[:, fam_idx[pool.family]]
                + self.idiosyncratic_weight * e_draws[:, k]
            )
            xi[:, k] = (z_k < threshold).astype(np.float64)

        return xi

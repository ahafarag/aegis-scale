"""Kingman-Sakasegawa G/G/c heavy-traffic queueing bounds and capacity bisection.

Maps upper arrival rate lambda (req/s), per-vCPU service rate mu (req/s/vCPU),
inter-arrival squared CV c_a^2, and service squared CV c_s^2 to stationary
queueing wait quantiles W_{q, 1-delta}, and solves for the minimum safe vCPU
capacity C_req guaranteeing P(W_q <= tau_SLA) >= 1 - delta.
"""

from __future__ import annotations

import math


def ggc_wait_probability(arrival_rate_rps: float, vcpus: float, per_vcpu_mu: float) -> float:
    """Sakasegawa approximation for P(W_q > 0) in a multi-server G/G/c system.

    Reduces to exact M/M/1 P(W_q > 0) = rho when c = 1.
    """
    if arrival_rate_rps <= 0.0:
        return 0.0
    if vcpus <= 0.0 or per_vcpu_mu <= 0.0:
        return 1.0
    rho = arrival_rate_rps / (vcpus * per_vcpu_mu)
    if rho >= 1.0:
        return 1.0
    exponent = math.sqrt(2.0 * (vcpus + 1.0)) - 1.0
    return float(rho**exponent)


def ggc_mean_wait_seconds(
    arrival_rate_rps: float,
    vcpus: float,
    per_vcpu_mu: float,
    arrival_cv2: float = 1.0,
    service_cv2: float = 1.0,
) -> float:
    """Kingman-Sakasegawa heavy-traffic approximation for mean queueing delay E[W_q] (seconds).

    Exact for M/M/1 (c=1, c_a^2=1, c_s^2=1), M/G/1 (Pollaczek-Khinchine), and
    asymptotically exact for G/G/c as rho -> 1.
    """
    if arrival_rate_rps <= 0.0:
        return 0.0
    if vcpus <= 0.0 or per_vcpu_mu <= 0.0:
        return math.inf
    rho = arrival_rate_rps / (vcpus * per_vcpu_mu)
    if rho >= 1.0:
        return math.inf

    variability = 0.5 * (max(0.0, arrival_cv2) + max(0.0, service_cv2))
    if variability == 0.0:
        return 0.0
    exponent = math.sqrt(2.0 * (vcpus + 1.0)) - 1.0
    return float(variability * (rho**exponent) / (vcpus * per_vcpu_mu * (1.0 - rho)))


def ggc_quantile_wait_seconds(
    arrival_rate_rps: float,
    vcpus: float,
    per_vcpu_mu: float,
    quantile: float = 0.99,
    arrival_cv2: float = 1.0,
    service_cv2: float = 1.0,
) -> float:
    """Stationary queueing wait quantile W_{q, quantile} in seconds.

    Combines Sakasegawa delay probability P_wait(c) with Kingman's exponential
    heavy-traffic conditional waiting-time law:
        P(W_q > w) = P_wait(c) * exp(-w / E[W_q | W_q > 0])
    where E[W_q | W_q > 0] = E[W_q] / P_wait(c) = ((c_a^2 + c_s^2)/2) / (c * mu * (1 - rho)).

    At c = 1, c_a^2 = c_s^2 = 1, this yields the exact closed-form M/M/1 quantile:
        W_{q, 1-delta} = max(0, ln(rho / delta) / (mu * (1 - rho))).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must lie strictly in (0, 1)")
    if arrival_rate_rps <= 0.0:
        return 0.0
    if vcpus <= 0.0 or per_vcpu_mu <= 0.0:
        return math.inf

    rho = arrival_rate_rps / (vcpus * per_vcpu_mu)
    if rho >= 1.0:
        return math.inf

    variability = 0.5 * (max(0.0, arrival_cv2) + max(0.0, service_cv2))
    if variability == 0.0:
        return 0.0

    delta = 1.0 - quantile
    p_wait = ggc_wait_probability(arrival_rate_rps, vcpus, per_vcpu_mu)
    if p_wait <= delta:
        return 0.0

    cond_mean_wait = variability / (vcpus * per_vcpu_mu * (1.0 - rho))
    return float(cond_mean_wait * math.log(p_wait / delta))


def required_safe_vcpus(
    arrival_rate_rps: float,
    per_vcpu_mu: float,
    sla_wait_ms: float = 25.0,
    quantile: float = 0.99,
    arrival_cv2: float = 1.6,
    service_cv2: float = 1.2,
    min_vcpus: float = 1.0,
    tol: float = 1e-6,
) -> float:
    """Compute exact continuous minimum vCPU capacity C_req such that W_{q, quantile}(C_req) <= sla_wait_ms.

    Because W_{q, quantile}(c) is strictly decreasing and continuous for c > lambda/mu,
    bisection converges globally to machine tolerance in O(log(1/tol)) evaluations.
    """
    if per_vcpu_mu <= 0.0:
        raise ValueError("per_vcpu_mu must be strictly positive")
    if sla_wait_ms <= 0.0:
        raise ValueError("sla_wait_ms must be strictly positive")
    if arrival_rate_rps <= 0.0:
        return float(min_vcpus)

    target_sec = sla_wait_ms / 1000.0
    stability_floor = arrival_rate_rps / per_vcpu_mu

    lo = max(stability_floor * (1.0 + 1e-9), min_vcpus)
    if ggc_quantile_wait_seconds(arrival_rate_rps, lo, per_vcpu_mu, quantile, arrival_cv2, service_cv2) <= target_sec:
        return float(lo)

    hi = max(lo * 2.0, stability_floor + 4.0)
    while ggc_quantile_wait_seconds(arrival_rate_rps, hi, per_vcpu_mu, quantile, arrival_cv2, service_cv2) > target_sec:
        hi *= 2.0
        if hi > 1e8:
            break

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (hi - lo) <= tol:
            return float(hi)
        w = ggc_quantile_wait_seconds(arrival_rate_rps, mid, per_vcpu_mu, quantile, arrival_cv2, service_cv2)
        if w > target_sec:
            lo = mid
        else:
            hi = mid

    return float(hi)

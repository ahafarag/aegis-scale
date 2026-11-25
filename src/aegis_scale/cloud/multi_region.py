"""Multi-region AWS EC2 On-Demand and Spot compute catalogs and scalable benchmark generator.

Provides calibrated regional pricing and empirical Spot interruption hazard models across
us-east-1 (N. Virginia), eu-west-1 (Ireland), and ap-northeast-1 (Tokyo), including
memory-optimized Graviton3 (r7g) pools for 2D Kubernetes pod bin-packing, plus a
deterministic synthetic catalog generator for large-scale MILP scalability benchmarks.
"""

from __future__ import annotations

import numpy as np

from aegis_scale.config import ComputePool

SUPPORTED_REGIONS: tuple[str, ...] = ("us-east-1", "eu-west-1", "ap-northeast-1")


def _us_east_1_pools() -> list[ComputePool]:
    """Reference N. Virginia catalog across us-east-1a, us-east-1b, and us-east-1c."""
    azs = ["us-east-1a", "us-east-1b", "us-east-1c"]
    pools: list[ComputePool] = [
        ComputePool(
            pool_id="od-m7g.xlarge-us-east-1a",
            instance_type="m7g.xlarge",
            family="m7g",
            az="us-east-1a",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.1632,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-c7g.xlarge-us-east-1b",
            instance_type="c7g.xlarge",
            family="c7g",
            az="us-east-1b",
            market="on-demand",
            vcpu=4,
            memory_gib=8.0,
            hourly_price_usd=0.1445,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-r7g.xlarge-us-east-1b",
            instance_type="r7g.xlarge",
            family="r7g",
            az="us-east-1b",
            market="on-demand",
            vcpu=4,
            memory_gib=32.0,
            hourly_price_usd=0.2142,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-m6i.xlarge-us-east-1c",
            instance_type="m6i.xlarge",
            family="m6i",
            az="us-east-1c",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.1920,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
    ]

    spot_specs = [
        ("c6a.xlarge", "c6a", 4, 8.0, 0.0410, 0.075),
        ("c7g.xlarge", "c7g", 4, 8.0, 0.0495, 0.045),
        ("m7g.xlarge", "m7g", 4, 16.0, 0.0560, 0.040),
        ("r7g.xlarge", "r7g", 4, 32.0, 0.0672, 0.036),
        ("m6a.xlarge", "m6a", 4, 16.0, 0.0540, 0.050),
        ("c6i.xlarge", "c6i", 4, 8.0, 0.0590, 0.032),
        ("m6i.2xlarge", "m6i", 8, 32.0, 0.1240, 0.028),
    ]

    az_price_mult = {"us-east-1a": 1.00, "us-east-1b": 0.98, "us-east-1c": 1.03}
    az_hazard_offset = {"us-east-1a": 0.005, "us-east-1b": 0.000, "us-east-1c": -0.004}

    for inst_type, fam, vcpu, mem, base_price, base_prob in spot_specs:
        for az in azs:
            pools.append(
                ComputePool(
                    pool_id=f"spot-{inst_type}-{az}",
                    instance_type=inst_type,
                    family=fam,
                    az=az,
                    market="spot",
                    vcpu=vcpu,
                    memory_gib=mem,
                    hourly_price_usd=round(base_price * az_price_mult[az], 5),
                    base_interruption_prob=round(max(0.005, base_prob + az_hazard_offset[az]), 4),
                    max_instances=80,
                )
            )

    return pools


def _eu_west_1_pools() -> list[ComputePool]:
    """Ireland catalog (eu-west-1a, b, c): ~10% higher OD pricing, lower baseline Spot hazard."""
    azs = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
    pools: list[ComputePool] = [
        ComputePool(
            pool_id="od-m7g.xlarge-eu-west-1a",
            instance_type="m7g.xlarge",
            family="m7g",
            az="eu-west-1a",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.1808,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-c7g.xlarge-eu-west-1b",
            instance_type="c7g.xlarge",
            family="c7g",
            az="eu-west-1b",
            market="on-demand",
            vcpu=4,
            memory_gib=8.0,
            hourly_price_usd=0.1592,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-r7g.xlarge-eu-west-1b",
            instance_type="r7g.xlarge",
            family="r7g",
            az="eu-west-1b",
            market="on-demand",
            vcpu=4,
            memory_gib=32.0,
            hourly_price_usd=0.2368,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-m6i.xlarge-eu-west-1c",
            instance_type="m6i.xlarge",
            family="m6i",
            az="eu-west-1c",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.2112,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
    ]

    # Distinct European Spot discounts (64%-74% off regional OD) and lower baseline recall rates
    spot_specs = [
        ("c6a.xlarge", "c6a", 4, 8.0, 0.0462, 0.058),
        ("c7g.xlarge", "c7g", 4, 8.0, 0.0535, 0.034),
        ("m7g.xlarge", "m7g", 4, 16.0, 0.0615, 0.030),
        ("r7g.xlarge", "r7g", 4, 32.0, 0.0745, 0.027),
        ("m6a.xlarge", "m6a", 4, 16.0, 0.0598, 0.038),
        ("c6i.xlarge", "c6i", 4, 8.0, 0.0648, 0.025),
        ("m6i.2xlarge", "m6i", 8, 32.0, 0.1365, 0.021),
    ]

    az_price_mult = {"eu-west-1a": 1.01, "eu-west-1b": 0.99, "eu-west-1c": 1.02}
    az_hazard_offset = {"eu-west-1a": 0.003, "eu-west-1b": -0.001, "eu-west-1c": -0.003}

    for inst_type, fam, vcpu, mem, base_price, base_prob in spot_specs:
        for az in azs:
            pools.append(
                ComputePool(
                    pool_id=f"spot-{inst_type}-{az}",
                    instance_type=inst_type,
                    family=fam,
                    az=az,
                    market="spot",
                    vcpu=vcpu,
                    memory_gib=mem,
                    hourly_price_usd=round(base_price * az_price_mult[az], 5),
                    base_interruption_prob=round(max(0.004, base_prob + az_hazard_offset[az]), 4),
                    max_instances=80,
                )
            )

    return pools


def _ap_northeast_1_pools() -> list[ComputePool]:
    """Tokyo catalog (ap-northeast-1a, c, d): ~13% higher OD pricing, tighter cross-AZ Spot supply."""
    azs = ["ap-northeast-1a", "ap-northeast-1c", "ap-northeast-1d"]
    pools: list[ComputePool] = [
        ComputePool(
            pool_id="od-m7g.xlarge-ap-northeast-1a",
            instance_type="m7g.xlarge",
            family="m7g",
            az="ap-northeast-1a",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.1856,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-c7g.xlarge-ap-northeast-1c",
            instance_type="c7g.xlarge",
            family="c7g",
            az="ap-northeast-1c",
            market="on-demand",
            vcpu=4,
            memory_gib=8.0,
            hourly_price_usd=0.1636,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-r7g.xlarge-ap-northeast-1c",
            instance_type="r7g.xlarge",
            family="r7g",
            az="ap-northeast-1c",
            market="on-demand",
            vcpu=4,
            memory_gib=32.0,
            hourly_price_usd=0.2428,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
        ComputePool(
            pool_id="od-m6i.xlarge-ap-northeast-1d",
            instance_type="m6i.xlarge",
            family="m6i",
            az="ap-northeast-1d",
            market="on-demand",
            vcpu=4,
            memory_gib=16.0,
            hourly_price_usd=0.2176,
            base_interruption_prob=0.0,
            max_instances=150,
        ),
    ]

    # Tighter Tokyo Spot discounts (58%-70% off regional OD) and tighter, uniform cross-AZ hazard
    spot_specs = [
        ("c6a.xlarge", "c6a", 4, 8.0, 0.0520, 0.086),
        ("c7g.xlarge", "c7g", 4, 8.0, 0.0595, 0.056),
        ("m7g.xlarge", "m7g", 4, 16.0, 0.0680, 0.051),
        ("r7g.xlarge", "r7g", 4, 32.0, 0.0825, 0.046),
        ("m6a.xlarge", "m6a", 4, 16.0, 0.0660, 0.062),
        ("c6i.xlarge", "c6i", 4, 8.0, 0.0715, 0.042),
        ("m6i.2xlarge", "m6i", 8, 32.0, 0.1510, 0.036),
    ]

    az_price_mult = {
        "ap-northeast-1a": 1.00,
        "ap-northeast-1c": 1.01,
        "ap-northeast-1d": 0.995,
    }
    # Tight cross-AZ hazard spread reflecting dense regional capacity correlation
    az_hazard_offset = {
        "ap-northeast-1a": 0.002,
        "ap-northeast-1c": 0.001,
        "ap-northeast-1d": -0.001,
    }

    for inst_type, fam, vcpu, mem, base_price, base_prob in spot_specs:
        for az in azs:
            pools.append(
                ComputePool(
                    pool_id=f"spot-{inst_type}-{az}",
                    instance_type=inst_type,
                    family=fam,
                    az=az,
                    market="spot",
                    vcpu=vcpu,
                    memory_gib=mem,
                    hourly_price_usd=round(base_price * az_price_mult[az], 5),
                    base_interruption_prob=round(max(0.006, base_prob + az_hazard_offset[az]), 4),
                    max_instances=80,
                )
            )

    return pools


def get_region_pools(region: str) -> list[ComputePool]:
    """Return calibrated On-Demand and Spot EC2 compute pools for a supported AWS region."""
    normalized = region.strip().lower()
    if normalized == "us-east-1":
        return _us_east_1_pools()
    if normalized == "eu-west-1":
        return _eu_west_1_pools()
    if normalized == "ap-northeast-1":
        return _ap_northeast_1_pools()
    raise ValueError(
        f"Unsupported region '{region}'. Supported regions: {', '.join(SUPPORTED_REGIONS)}"
    )


def generate_scaled_pool_catalog(n_pools: int, seed: int = 42) -> list[ComputePool]:
    """Generate a deterministic catalog of n_pools EC2 On-Demand and Spot pools for scalability tests.

    Supports benchmarking up to 250+ candidate pools across 6 Availability Zones, 12 instance
    families (general-purpose, compute-optimized, memory-optimized across Graviton, Intel, AMD),
    and 4 instance sizes (large through 4xlarge).
    """
    if n_pools < 2:
        raise ValueError("n_pools must be >= 2 to include both On-Demand and Spot pools")

    rng = np.random.default_rng(seed)
    azs = [f"us-east-1{s}" for s in ("a", "b", "c", "d", "e", "f")]

    # (family, mem_per_vcpu_gib, od_rate_per_vcpu_usd, base_spot_hazard)
    family_catalog = [
        ("c7g", 2.0, 0.0361, 0.044),
        ("c6i", 2.0, 0.0425, 0.034),
        ("c6a", 2.0, 0.0382, 0.068),
        ("c7i", 2.0, 0.0446, 0.039),
        ("m7g", 4.0, 0.0408, 0.038),
        ("m6i", 4.0, 0.0480, 0.030),
        ("m6a", 4.0, 0.0432, 0.052),
        ("m7i", 4.0, 0.0504, 0.036),
        ("m7a", 4.0, 0.0465, 0.048),
        ("r7g", 8.0, 0.0535, 0.035),
        ("r6i", 8.0, 0.0630, 0.029),
        ("r6a", 8.0, 0.0567, 0.046),
    ]
    sizes = [
        ("large", 2),
        ("xlarge", 4),
        ("2xlarge", 8),
        ("4xlarge", 16),
    ]

    # Build rich candidate combinations: (market, family_entry, size_entry, az)
    od_candidates: list[tuple[tuple[str, float, float, float], tuple[str, int], str]] = []
    spot_candidates: list[tuple[tuple[str, float, float, float], tuple[str, int], str]] = []
    for fam_entry in family_catalog:
        for size_entry in sizes:
            for az in azs:
                od_candidates.append((fam_entry, size_entry, az))
                spot_candidates.append((fam_entry, size_entry, az))

    rng.shuffle(od_candidates)
    rng.shuffle(spot_candidates)

    n_od = max(1, min(len(od_candidates), max(3, int(round(0.15 * n_pools)))))
    if n_od >= n_pools:
        n_od = 1
    n_spot = n_pools - n_od

    pools: list[ComputePool] = []

    for idx in range(n_od):
        (fam, mem_ratio, od_rate, _), (size_name, vcpu), az = od_candidates[idx % len(od_candidates)]
        inst_type = f"{fam}.{size_name}"
        suffix = "" if idx < len(od_candidates) else f"-g{idx // len(od_candidates)}"
        jitter = float(rng.uniform(0.98, 1.02))
        price = round(vcpu * od_rate * jitter, 5)
        pools.append(
            ComputePool(
                pool_id=f"od-{inst_type}-{az}{suffix}",
                instance_type=inst_type,
                family=fam,
                az=az,
                market="on-demand",
                vcpu=vcpu,
                memory_gib=float(vcpu * mem_ratio),
                hourly_price_usd=price,
                base_interruption_prob=0.0,
                max_instances=250,
            )
        )

    for idx in range(n_spot):
        (fam, mem_ratio, od_rate, base_haz), (size_name, vcpu), az = spot_candidates[
            idx % len(spot_candidates)
        ]
        inst_type = f"{fam}.{size_name}"
        suffix = "" if idx < len(spot_candidates) else f"-g{idx // len(spot_candidates)}"
        discount = float(rng.uniform(0.60, 0.76))
        spot_price = round(vcpu * od_rate * (1.0 - discount), 5)
        haz_jitter = float(rng.uniform(-0.010, 0.015))
        prob = round(float(np.clip(base_haz + haz_jitter, 0.008, 0.120)), 4)
        pools.append(
            ComputePool(
                pool_id=f"spot-{inst_type}-{az}{suffix}",
                instance_type=inst_type,
                family=fam,
                az=az,
                market="spot",
                vcpu=vcpu,
                memory_gib=float(vcpu * mem_ratio),
                hourly_price_usd=spot_price,
                base_interruption_prob=prob,
                max_instances=120,
            )
        )

    return pools

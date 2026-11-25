"""Realistic AWS EC2 On-Demand and Spot compute pool catalog (us-east-1 reference pricing)."""

from __future__ import annotations

from aegis_scale.config import ComputePool


def default_us_east_1_pools() -> list[ComputePool]:
    """Return a realistic multi-AZ, multi-family catalog of AWS EC2 On-Demand and Spot pools.

    Includes Graviton3 (m7g, c7g), Intel Ice Lake (m6i, c6i), and AMD EPYC (m6a, c6a)
    spread across us-east-1a, us-east-1b, and us-east-1c with realistic Spot discounts
    (62% to 76% off On-Demand) and differentiated empirical interruption probabilities.
    """
    azs = ["us-east-1a", "us-east-1b", "us-east-1c"]
    pools: list[ComputePool] = [
        # Reliable On-Demand anchor pools across AZs
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

    # Diversified Spot pools across 6 families and 3 Availability Zones
    spot_specs = [
        ("c6a.xlarge", "c6a", 4, 8.0, 0.0410, 0.075),   # Deepest discount, higher recall hazard
        ("c7g.xlarge", "c7g", 4, 8.0, 0.0495, 0.045),   # Graviton3 compute-optimized
        ("m7g.xlarge", "m7g", 4, 16.0, 0.0560, 0.040),  # Graviton3 general-purpose
        ("m6a.xlarge", "m6a", 4, 16.0, 0.0540, 0.050),  # AMD EPYC general-purpose
        ("c6i.xlarge", "c6i", 4, 8.0, 0.0590, 0.032),   # Intel compute-optimized, deeper pool
        ("m6i.2xlarge", "m6i", 8, 32.0, 0.1240, 0.028), # Larger Intel pool, lowest Spot hazard
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

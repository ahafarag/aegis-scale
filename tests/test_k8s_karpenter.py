"""Unit tests for Kubernetes 2D bin-packing, Karpenter v1 reconciliation, and multi-region catalogs."""

from __future__ import annotations

import numpy as np
import pytest

from aegis_scale.cloud.multi_region import (
    SUPPORTED_REGIONS,
    generate_scaled_pool_catalog,
    get_region_pools,
)
from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.k8s import (
    K8sAegisOperator,
    K8sClusterOverhead,
    KarpenterReconciler,
    PodResources,
    build_2d_effective_pools,
    pods_per_node,
    required_safe_pod_replicas,
)
from aegis_scale.math_core.cvar_milp import CVaRPortfolioOptimizer, PortfolioDecision
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel


def _extract_requirements_map(node_pool: dict) -> dict[str, list[str]]:
    reqs = node_pool["spec"]["template"]["spec"]["requirements"]
    return {r["key"]: r["values"] for r in reqs}


def test_2d_bin_packing_memory_heavy_pods_favor_r7g_and_m7g_over_c7g() -> None:
    """Verify 1 vCPU, 7 GiB pods expose c7g CPU stranding and steer MILP to r7g/m7g."""
    pools = get_region_pools("us-east-1")
    pool_by_id = {p.pool_id: p for p in pools}
    overhead = K8sClusterOverhead(
        daemonset_cpu_vcpu=0.25,
        daemonset_memory_gib=0.75,
        node_boot_seconds=65.0,
    )
    mem_pod = PodResources(
        name="vector-search-api",
        cpu_vcpu=1.0,
        memory_gib=7.0,
        target_rps_per_pod=80.0,
    )

    # c7g.xlarge (4 vCPU, 8 GiB): 8 - 0.75 = 7.25 GiB -> fits only 1 pod, stranding ~3 vCPUs
    c7g_pool = pool_by_id["spot-c7g.xlarge-us-east-1a"]
    m7g_pool = pool_by_id["spot-m7g.xlarge-us-east-1a"]
    r7g_pool = pool_by_id["spot-r7g.xlarge-us-east-1a"]

    assert pods_per_node(c7g_pool, mem_pod, overhead) == 1
    assert pods_per_node(m7g_pool, mem_pod, overhead) == 2
    assert pods_per_node(r7g_pool, mem_pod, overhead) == 3

    # Verify undersized nodes are filtered out cleanly
    tiny_pool = ComputePool(
        pool_id="od-t4g.medium-us-east-1a",
        instance_type="t4g.medium",
        family="t4g",
        az="us-east-1a",
        market="on-demand",
        vcpu=2,
        memory_gib=4.0,
        hourly_price_usd=0.0336,
    )
    assert pods_per_node(tiny_pool, mem_pod, overhead) == 0
    eff_pools, cap_map = build_2d_effective_pools([*pools, tiny_pool], mem_pod, overhead)
    assert "od-t4g.medium-us-east-1a" not in cap_map
    assert len(eff_pools) == len(pools)

    # Run full K8s operator on memory-heavy pod workload
    operator = K8sAegisOperator(
        pools=pools,
        pod=mem_pod,
        overhead=overhead,
        seed=101,
    )
    report = operator.run_cycle(observed_rps=2400.0)

    c7g_pods = 0
    r7g_m7g_pods = 0
    total_pods = 0
    for pid, node_count in report.portfolio.allocations.items():
        fam = pool_by_id[pid].family
        pod_slots = node_count * cap_map[pid]
        total_pods += pod_slots
        if fam == "c7g":
            c7g_pods += pod_slots
        elif fam in ("r7g", "m7g"):
            r7g_m7g_pods += pod_slots

    assert total_pods == report.total_schedulable_pods
    assert total_pods >= report.required_pod_replicas
    assert c7g_pods == 0
    assert r7g_m7g_pods >= int(0.60 * total_pods)
    assert report.portfolio.cvar_shortfall_vcpu == 0.0


def test_karpenter_v1_nodepool_ec2nodeclass_and_scale_manifests() -> None:
    """Validate karpenter.sh/v1 NodePool, karpenter.k8s.aws/v1 EC2NodeClass, and Scale specs."""
    pools = get_region_pools("us-east-1")
    pod = PodResources(
        name="checkout-gateway",
        cpu_vcpu=1.0,
        memory_gib=3.5,
        target_rps_per_pod=110.0,
    )
    reconciler = KarpenterReconciler(
        cluster_name="aegis-prod-cluster",
        namespace="commerce",
        node_class_name="aegis-hardened-al2023",
    )
    operator = K8sAegisOperator(
        pools=pools,
        pod=pod,
        reconciler=reconciler,
        interval_sec=30.0,
        seed=7,
    )

    r1 = operator.run_cycle(observed_rps=1200.0)
    # Steep upward surge triggers cold-start lead-time replica buffer
    r2 = operator.run_cycle(observed_rps=2100.0)
    assert r2.rps_ramp_rate_per_sec == pytest.approx((2100.0 - 1200.0) / 30.0)
    assert r2.required_pod_replicas > r1.required_pod_replicas

    bundle = r2.manifests
    assert len(bundle.all_manifests()) == 2 + len(bundle.node_pools)

    # 1. Validate EC2NodeClass security & AL2023 configuration
    nc = bundle.ec2_node_class
    assert nc["apiVersion"] == "karpenter.k8s.aws/v1"
    assert nc["kind"] == "EC2NodeClass"
    assert nc["metadata"]["name"] == "aegis-hardened-al2023"
    assert nc["spec"]["amiSelectorTerms"] == [{"alias": "al2023@latest"}]

    bdm = nc["spec"]["blockDeviceMappings"]
    assert len(bdm) == 1
    assert bdm[0]["ebs"]["volumeType"] == "gp3"
    assert bdm[0]["ebs"]["encrypted"] is True

    meta_opts = nc["spec"]["metadataOptions"]
    assert meta_opts == {
        "httpEndpoint": "enabled",
        "httpProtocolIPv6": "disabled",
        "httpPutResponseHopLimit": 2,
        "httpTokens": "required",
    }

    # 2. Validate NodePools
    assert len(bundle.node_pools) >= 2
    names = [np_doc["metadata"]["name"] for np_doc in bundle.node_pools]
    assert "aegis-ondemand-anchor" in names
    assert any(n.startswith("aegis-spot-") for n in names)

    pool_by_id = {p.pool_id: p for p in pools}
    expected_total_cpu = sum(
        cnt * pool_by_id[pid].vcpu for pid, cnt in r2.portfolio.allocations.items()
    )
    manifest_total_cpu = 0

    for np_doc in bundle.node_pools:
        assert np_doc["apiVersion"] == "karpenter.sh/v1"
        assert np_doc["kind"] == "NodePool"
        spec = np_doc["spec"]
        assert 1 <= spec["weight"] <= 100
        assert isinstance(spec["limits"]["cpu"], int) and spec["limits"]["cpu"] > 0
        assert spec["limits"]["memory"].endswith("Gi")
        mem_val = int(spec["limits"]["memory"][:-2])
        assert mem_val > 0
        manifest_total_cpu += spec["limits"]["cpu"]

        assert spec["disruption"] == {
            "consolidationPolicy": "WhenEmptyOrUnderutilized",
            "consolidateAfter": "60s",
        }
        node_ref = spec["template"]["spec"]["nodeClassRef"]
        assert node_ref == {
            "group": "karpenter.k8s.aws",
            "kind": "EC2NodeClass",
            "name": "aegis-hardened-al2023",
        }

        req_map = _extract_requirements_map(np_doc)
        assert set(req_map.keys()) == {
            "karpenter.sh/capacity-type",
            "topology.kubernetes.io/zone",
            "karpenter.k8s.aws/instance-family",
        }
        if np_doc["metadata"]["name"] == "aegis-ondemand-anchor":
            assert req_map["karpenter.sh/capacity-type"] == ["on-demand"]
        else:
            assert req_map["karpenter.sh/capacity-type"] == ["spot"]

    assert manifest_total_cpu == expected_total_cpu

    # Also verify consolidated single Spot NodePool mode (`aegis-spot-portfolio`)
    single_spot_reconciler = KarpenterReconciler(split_spot_by_az=False)
    single_bundle = single_spot_reconciler.reconcile(
        decision=r2.portfolio,
        pools=pools,
        pod=pod,
        required_replicas=r2.required_pod_replicas,
    )
    single_names = [d["metadata"]["name"] for d in single_bundle.node_pools]
    assert single_names == ["aegis-ondemand-anchor", "aegis-spot-portfolio"]

    # 3. Validate proactive autoscaling/v1 Scale subresource patch
    scale = bundle.scale_patch
    assert scale == {
        "apiVersion": "autoscaling/v1",
        "kind": "Scale",
        "metadata": {"name": "checkout-gateway", "namespace": "commerce"},
        "spec": {"replicas": r2.required_pod_replicas},
    }


def test_compromised_spot_pool_excluded_from_karpenter_nodepool_requirements() -> None:
    """Verify Spot pools flagged via preemption warning are excluded from NodePool requirements."""
    pools = get_region_pools("us-east-1")
    pod = PodResources(
        name="payment-authorizer",
        cpu_vcpu=1.0,
        memory_gib=7.0,
        target_rps_per_pod=90.0,
    )
    operator = K8sAegisOperator(pools=pools, pod=pod, seed=55)

    initial_report = operator.run_cycle(observed_rps=1800.0)
    pool_by_id = {p.pool_id: p for p in pools}

    # Identify an actively allocated Spot pool to flag with a 2-minute preemption notice
    allocated_spot_ids = [
        pid for pid in initial_report.portfolio.allocations if pool_by_id[pid].market == "spot"
    ]
    assert len(allocated_spot_ids) >= 1
    target_pool = pool_by_id[allocated_spot_ids[0]]

    flagged = operator.flag_pool_preemption_warning(
        instance_type=target_pool.instance_type,
        az=target_pool.az,
    )
    assert flagged == [target_pool.pool_id]

    warned_report = operator.run_cycle(observed_rps=1800.0)
    assert target_pool.pool_id in warned_report.compromised_pools
    assert target_pool.pool_id not in warned_report.portfolio.allocations

    # Verify no Spot NodePool permits (target_pool.family, target_pool.az)
    for np_doc in warned_report.manifests.node_pools:
        req_map = _extract_requirements_map(np_doc)
        if req_map["karpenter.sh/capacity-type"] == ["spot"]:
            if target_pool.az in req_map["topology.kubernetes.io/zone"]:
                assert target_pool.family not in req_map["karpenter.k8s.aws/instance-family"]

    # Also test direct KarpenterReconciler defense-in-depth when decision contains compromised pool
    synthetic_decision = PortfolioDecision(
        allocations={
            "od-r7g.xlarge-us-east-1b": 2,
            "spot-r7g.xlarge-us-east-1a": 3,
            "spot-m7g.xlarge-us-east-1a": 2,
        },
        required_vcpus=15.0,
        total_provisioned_vcpus=19,
        on_demand_vcpus=6,
        spot_vcpus=13,
        hourly_cost_usd=0.74,
        all_on_demand_baseline_cost_usd=1.07,
        estimated_savings_pct=30.8,
        var_shortfall_vcpu=0.0,
        cvar_shortfall_vcpu=0.0,
        active_spot_pools=2,
        solve_time_ms=4.0,
    )
    direct_bundle = operator.reconciler.reconcile(
        decision=synthetic_decision,
        pools=pools,
        pod=pod,
        required_replicas=15,
        compromised_pools=["spot-r7g.xlarge-us-east-1a"],
    )
    spot_az1a = next(
        d for d in direct_bundle.node_pools if d["metadata"]["name"] == "aegis-spot-us-east-1a"
    )
    az1a_reqs = _extract_requirements_map(spot_az1a)
    assert "r7g" not in az1a_reqs["karpenter.k8s.aws/instance-family"]
    assert az1a_reqs["karpenter.k8s.aws/instance-family"] == ["m7g"]

    operator.clear_pool_warnings()
    cleared_report = operator.run_cycle(observed_rps=1800.0)
    assert cleared_report.compromised_pools == []


def test_cold_start_surge_buffer_in_pod_sizing() -> None:
    """Verify node boot lead-time buffer adds replicas on positive surge and ignores negative ramps."""
    pod = PodResources(
        name="stream-ingest",
        cpu_vcpu=2.0,
        memory_gib=4.0,
        target_rps_per_pod=150.0,
    )
    steady = required_safe_pod_replicas(
        design_rps=1500.0,
        pod=pod,
        cold_start_surge_rate_rps_per_sec=0.0,
        node_boot_seconds=65.0,
    )
    surging = required_safe_pod_replicas(
        design_rps=1500.0,
        pod=pod,
        cold_start_surge_rate_rps_per_sec=12.0,
        node_boot_seconds=65.0,
    )
    cooling = required_safe_pod_replicas(
        design_rps=1500.0,
        pod=pod,
        cold_start_surge_rate_rps_per_sec=-15.0,
        node_boot_seconds=65.0,
    )
    assert surging > steady
    assert cooling == steady


def test_all_regional_catalogs_and_scaled_210_pool_catalog_solve_cleanly() -> None:
    """Validate us-east-1, eu-west-1, ap-northeast-1, and 210-pool synthetic scalability catalog."""
    assert SUPPORTED_REGIONS == ("us-east-1", "eu-west-1", "ap-northeast-1")
    with pytest.raises(ValueError, match="Unsupported region"):
        get_region_pools("sa-east-1")

    region_catalogs = {reg: get_region_pools(reg) for reg in SUPPORTED_REGIONS}

    # Verify regional pricing and hazard structure relative to us-east-1
    us_od_m7g = next(
        p.hourly_price_usd
        for p in region_catalogs["us-east-1"]
        if p.market == "on-demand" and p.instance_type == "m7g.xlarge"
    )
    eu_od_m7g = next(
        p.hourly_price_usd
        for p in region_catalogs["eu-west-1"]
        if p.market == "on-demand" and p.instance_type == "m7g.xlarge"
    )
    ap_od_m7g = next(
        p.hourly_price_usd
        for p in region_catalogs["ap-northeast-1"]
        if p.market == "on-demand" and p.instance_type == "m7g.xlarge"
    )
    assert 1.08 <= eu_od_m7g / us_od_m7g <= 1.14
    assert 1.08 <= ap_od_m7g / us_od_m7g <= 1.15

    us_spot_haz = [p.base_interruption_prob for p in region_catalogs["us-east-1"] if p.market == "spot"]
    eu_spot_haz = [p.base_interruption_prob for p in region_catalogs["eu-west-1"] if p.market == "spot"]
    assert float(np.mean(eu_spot_haz)) < float(np.mean(us_spot_haz))

    pod = PodResources(
        name="regional-edge",
        cpu_vcpu=1.0,
        memory_gib=6.5,
        target_rps_per_pod=100.0,
    )
    for region, pools in region_catalogs.items():
        assert any(p.family == "r7g" for p in pools), f"{region} missing r7g pools"
        azs = {p.az for p in pools}
        assert len(azs) == 3
        assert all(az.startswith(region) for az in azs)

        op = K8sAegisOperator(pools=pools, pod=pod, seed=42)
        rep = op.run_cycle(observed_rps=1600.0)
        assert rep.total_schedulable_pods >= rep.required_pod_replicas
        assert rep.portfolio.cvar_shortfall_vcpu == 0.0
        assert rep.portfolio.estimated_savings_pct > 15.0

    # Validate 210-pool scalability catalog
    scaled_pools = generate_scaled_pool_catalog(210, seed=42)
    assert len(scaled_pools) == 210
    assert len({p.pool_id for p in scaled_pools}) == 210
    assert any(p.market == "on-demand" for p in scaled_pools)
    assert any(p.market == "spot" for p in scaled_pools)

    policy = SLAPolicy(num_scenarios=300)
    hazard = CorrelatedSpotHazardModel(
        az_correlation=policy.az_correlation,
        family_correlation=policy.family_correlation,
    )
    scenarios = hazard.sample_scenarios(scaled_pools, num_scenarios=policy.num_scenarios, seed=99)
    optimizer = CVaRPortfolioOptimizer(policy=policy)
    decision = optimizer.solve(pools=scaled_pools, required_vcpus=240.0, scenarios=scenarios)

    assert decision.total_provisioned_vcpus >= 240
    assert decision.cvar_shortfall_vcpu == 0.0
    assert decision.estimated_savings_pct > 20.0

"""Kubernetes Karpenter v1 manifest synthesis and proactive AegisScale K8s operator.

Translates optimal CVaR MILP fleet decisions into declarative `karpenter.sh/v1` NodePools,
hardened `karpenter.k8s.aws/v1` EC2NodeClass specifications (AL2023 + IMDSv2 hop-limit 2 +
encrypted gp3 EBS), and proactive `autoscaling/v1` Scale subresource patches that pre-warm
nodes ahead of steep RPS surges.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import math
from typing import Any

from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.k8s.bin_packing import (
    K8sClusterOverhead,
    PodResources,
    build_2d_effective_pools,
    required_safe_pod_replicas,
)
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor, ConformalStepResult
from aegis_scale.math_core.cvar_milp import (
    CVaRPortfolioOptimizer,
    InfeasibleCapacityError,
    PortfolioDecision,
)
from aegis_scale.math_core.hazard_copula import CorrelatedSpotHazardModel
from aegis_scale.math_core.queuing import ggc_quantile_wait_seconds


@dataclass(frozen=True)
class KarpenterManifestBundle:
    """Synthesized Kubernetes and Karpenter v1 resources for a single reconciliation cycle."""

    node_pools: list[dict[str, Any]]
    ec2_node_class: dict[str, Any]
    scale_patch: dict[str, Any]

    def all_manifests(self) -> list[dict[str, Any]]:
        """Return ordered list of EC2NodeClass, NodePools, and Scale resources."""
        return [self.ec2_node_class, *self.node_pools, self.scale_patch]


@dataclass(frozen=True)
class K8sControlCycleReport:
    """Complete telemetry, 2D sizing, CVaR portfolio, and Karpenter manifests for one K8s step."""

    conformal: ConformalStepResult
    observed_rps: float
    design_rps: float
    rps_ramp_rate_per_sec: float
    required_pod_replicas: int
    total_schedulable_pods: int
    expected_p99_queue_wait_ms: float
    portfolio: PortfolioDecision
    manifests: KarpenterManifestBundle
    compromised_pools: list[str]


class KarpenterReconciler:
    """Synthesizes `karpenter.sh/v1` NodePools, `EC2NodeClass`, and `autoscaling/v1` Scale specs."""

    def __init__(
        self,
        cluster_name: str = "aegis-eks-prod",
        namespace: str = "default",
        node_class_name: str = "aegis-al2023-default",
        iam_role_name: str = "AegisScaleKarpenterNodeRole",
        root_volume_size_gi: int = 50,
        split_spot_by_az: bool = True,
    ) -> None:
        if root_volume_size_gi < 8:
            raise ValueError("root_volume_size_gi must be >= 8 GiB")
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.node_class_name = node_class_name
        self.iam_role_name = iam_role_name
        self.root_volume_size_gi = int(root_volume_size_gi)
        self.split_spot_by_az = bool(split_spot_by_az)

    def build_ec2_node_class(self) -> dict[str, Any]:
        """Generate hardened `karpenter.k8s.aws/v1` EC2NodeClass with AL2023, gp3, and IMDSv2."""
        return {
            "apiVersion": "karpenter.k8s.aws/v1",
            "kind": "EC2NodeClass",
            "metadata": {
                "name": self.node_class_name,
                "labels": {
                    "app.kubernetes.io/managed-by": "aegis-scale",
                },
            },
            "spec": {
                "role": self.iam_role_name,
                "amiSelectorTerms": [
                    {"alias": "al2023@latest"},
                ],
                "subnetSelectorTerms": [
                    {"tags": {"karpenter.sh/discovery": self.cluster_name}},
                ],
                "securityGroupSelectorTerms": [
                    {"tags": {"karpenter.sh/discovery": self.cluster_name}},
                ],
                "blockDeviceMappings": [
                    {
                        "deviceName": "/dev/xvda",
                        "ebs": {
                            "volumeSize": f"{self.root_volume_size_gi}Gi",
                            "volumeType": "gp3",
                            "iops": 3000,
                            "throughput": 125,
                            "encrypted": True,
                            "deleteOnTermination": True,
                        },
                    }
                ],
                "metadataOptions": {
                    "httpEndpoint": "enabled",
                    "httpProtocolIPv6": "disabled",
                    "httpPutResponseHopLimit": 2,
                    "httpTokens": "required",
                },
            },
        }

    def build_scale_patch(self, pod: PodResources, required_replicas: int) -> dict[str, Any]:
        """Generate proactive `autoscaling/v1` Scale patch dict ahead of traffic bursts."""
        if required_replicas < 1:
            raise ValueError("required_replicas must be >= 1")
        return {
            "apiVersion": "autoscaling/v1",
            "kind": "Scale",
            "metadata": {
                "name": pod.name,
                "namespace": self.namespace,
            },
            "spec": {
                "replicas": int(required_replicas),
            },
        }

    def build_node_pools(
        self,
        decision: PortfolioDecision,
        pools: list[ComputePool],
        pod: PodResources,
        compromised_pools: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Synthesize `karpenter.sh/v1` NodePool manifests from non-compromised allocations."""
        pool_by_id = {p.pool_id: p for p in pools}
        compromised_ids = set(compromised_pools or ())
        compromised_spot_pairs = {
            (pool_by_id[pid].family, pool_by_id[pid].az)
            for pid in compromised_ids
            if pid in pool_by_id and pool_by_id[pid].market == "spot"
        }

        od_entries: list[tuple[ComputePool, int]] = []
        spot_entries: list[tuple[ComputePool, int]] = []

        for pool_id, count in decision.allocations.items():
            if count <= 0 or pool_id not in pool_by_id or pool_id in compromised_ids:
                continue
            pool = pool_by_id[pool_id]
            if pool.market == "on-demand":
                od_entries.append((pool, int(count)))
            else:
                if (pool.family, pool.az) in compromised_spot_pairs:
                    continue
                spot_entries.append((pool, int(count)))

        groups: list[tuple[str, str, list[tuple[ComputePool, int]]]] = []
        if od_entries:
            groups.append(("aegis-ondemand-anchor", "on-demand", od_entries))

        if spot_entries:
            if self.split_spot_by_az:
                spot_azs = sorted({p.az for p, _ in spot_entries})
                for az in spot_azs:
                    az_entries = [(p, c) for p, c in spot_entries if p.az == az]
                    if az_entries:
                        groups.append((f"aegis-spot-{az}", "spot", az_entries))
            else:
                # Filter out any family compromised in any target Spot AZ to prevent cross-product leak
                active_spot_azs = {p.az for p, _ in spot_entries}
                blocked_families = {
                    fam for fam, az in compromised_spot_pairs if az in active_spot_azs
                }
                safe_spot = [(p, c) for p, c in spot_entries if p.family not in blocked_families]
                if safe_spot:
                    groups.append(("aegis-spot-portfolio", "spot", safe_spot))

        group_cpus = [sum(c * p.vcpu for p, c in entries) for _, _, entries in groups]
        total_cpu = max(1, sum(group_cpus))

        manifests: list[dict[str, Any]] = []
        for (name, cap_type, entries), grp_cpu in zip(groups, group_cpus):
            grp_mem_gi = max(1, int(math.ceil(sum(c * p.memory_gib for p, c in entries))))
            weight = max(1, min(100, int(round(100.0 * float(grp_cpu) / float(total_cpu)))))
            zones = sorted({p.az for p, _ in entries})
            families = sorted({p.family for p, _ in entries})

            manifests.append(
                {
                    "apiVersion": "karpenter.sh/v1",
                    "kind": "NodePool",
                    "metadata": {
                        "name": name,
                        "labels": {
                            "app.kubernetes.io/managed-by": "aegis-scale",
                            "aegis-scale.io/workload": pod.name,
                            "aegis-scale.io/capacity-tier": cap_type,
                        },
                    },
                    "spec": {
                        "weight": weight,
                        "limits": {
                            "cpu": int(grp_cpu),
                            "memory": f"{grp_mem_gi}Gi",
                        },
                        "disruption": {
                            "consolidationPolicy": "WhenEmptyOrUnderutilized",
                            "consolidateAfter": "60s",
                        },
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app.kubernetes.io/managed-by": "aegis-scale",
                                    "aegis-scale.io/workload": pod.name,
                                },
                            },
                            "spec": {
                                "nodeClassRef": {
                                    "group": "karpenter.k8s.aws",
                                    "kind": "EC2NodeClass",
                                    "name": self.node_class_name,
                                },
                                "expireAfter": "720h",
                                "requirements": [
                                    {
                                        "key": "karpenter.sh/capacity-type",
                                        "operator": "In",
                                        "values": [cap_type],
                                    },
                                    {
                                        "key": "topology.kubernetes.io/zone",
                                        "operator": "In",
                                        "values": zones,
                                    },
                                    {
                                        "key": "karpenter.k8s.aws/instance-family",
                                        "operator": "In",
                                        "values": families,
                                    },
                                ],
                            },
                        },
                    },
                }
            )

        return manifests

    def reconcile(
        self,
        decision: PortfolioDecision,
        pools: list[ComputePool],
        pod: PodResources,
        required_replicas: int,
        compromised_pools: Collection[str] | None = None,
    ) -> KarpenterManifestBundle:
        """Synthesize complete NodePool, EC2NodeClass, and Scale manifests for a portfolio."""
        node_pools = self.build_node_pools(
            decision=decision,
            pools=pools,
            pod=pod,
            compromised_pools=compromised_pools,
        )
        ec2_node_class = self.build_ec2_node_class()
        scale_patch = self.build_scale_patch(pod=pod, required_replicas=required_replicas)
        return KarpenterManifestBundle(
            node_pools=node_pools,
            ec2_node_class=ec2_node_class,
            scale_patch=scale_patch,
        )


class K8sAegisOperator:
    """Closed-loop Kubernetes/Karpenter operator with ACI, 2D pod sizing, and CVaR MILP."""

    def __init__(
        self,
        pools: list[ComputePool],
        pod: PodResources,
        policy: SLAPolicy | None = None,
        overhead: K8sClusterOverhead | None = None,
        reconciler: KarpenterReconciler | None = None,
        interval_sec: float = 60.0,
        seed: int = 42,
    ) -> None:
        if not pools:
            raise ValueError("pools must contain at least one ComputePool")
        if interval_sec <= 0.0:
            raise ValueError("interval_sec must be > 0")

        self.pools = list(pools)
        self.pod = pod
        self.policy = policy or SLAPolicy()
        self.overhead = overhead or K8sClusterOverhead()
        self.reconciler = reconciler or KarpenterReconciler()
        self.interval_sec = float(interval_sec)
        self.seed = int(seed)

        self.effective_pools, self.pods_per_node_map = build_2d_effective_pools(
            pools=self.pools,
            pod=self.pod,
            overhead=self.overhead,
        )
        if not self.effective_pools:
            raise InfeasibleCapacityError(
                f"No candidate EC2 pool can fit pod '{self.pod.name}' "
                f"({self.pod.cpu_vcpu} vCPU, {self.pod.memory_gib} GiB) after DaemonSet overhead"
            )

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
        self._last_observed_rps: float | None = None
        self._cycle_index = 0

    def flag_pool_preemption_warning(
        self,
        instance_type: str,
        az: str,
        elevated_prob: float = 0.98,
    ) -> list[str]:
        """Flag Spot pool(s) matching (instance_type, az) upon AWS interruption/rebalance notice."""
        flagged: list[str] = []
        for pool in self.pools:
            if pool.market == "spot" and pool.instance_type == instance_type and pool.az == az:
                self.elevated_hazards[pool.pool_id] = max(0.5, min(0.999, float(elevated_prob)))
                flagged.append(pool.pool_id)
        return flagged

    def clear_pool_warnings(self) -> None:
        """Clear transient Spot preemption warnings after cool-down."""
        self.elevated_hazards.clear()

    def run_cycle(self, observed_rps: float) -> K8sControlCycleReport:
        """Execute one operator cycle: ACI -> 2D surge-buffered pods -> CVaR MILP -> Karpenter v1."""
        obs = max(0.0, float(observed_rps))
        if self._last_observed_rps is None:
            ramp_rate = 0.0
        else:
            ramp_rate = (obs - self._last_observed_rps) / self.interval_sec
        self._last_observed_rps = obs

        conformal_res = self.predictor.step(obs)
        _, next_upper_rps = self.predictor.predict_upper_bound()
        design_rps = max(next_upper_rps, obs)

        buffered_arrival_rps = design_rps + max(0.0, ramp_rate) * self.overhead.node_boot_seconds
        req_replicas = required_safe_pod_replicas(
            design_rps=design_rps,
            pod=self.pod,
            sla_wait_ms=self.policy.p99_queue_delay_sla_ms,
            arrival_cv2=self.policy.arrival_cv2,
            service_cv2=self.policy.service_cv2,
            cold_start_surge_rate_rps_per_sec=max(0.0, ramp_rate),
            node_boot_seconds=self.overhead.node_boot_seconds,
        )

        # Exclude actively recalled/compromised Spot pools from candidate scheduling pools
        candidate_effective_pools = [
            p for p in self.effective_pools if p.pool_id not in self.elevated_hazards
        ]
        if not candidate_effective_pools:
            raise InfeasibleCapacityError("All schedulable candidate pools are marked compromised")

        scenarios = self.hazard_model.sample_scenarios(
            pools=candidate_effective_pools,
            num_scenarios=self.policy.num_scenarios,
            elevated_pool_hazards=self.elevated_hazards,
            seed=self.seed + self._cycle_index,
        )
        self._cycle_index += 1

        decision = self.optimizer.solve(
            pools=candidate_effective_pools,
            required_vcpus=float(req_replicas),
            scenarios=scenarios,
            previous_allocations=self.current_allocations,
        )
        self.current_allocations = dict(decision.allocations)

        compromised_list = sorted(self.elevated_hazards.keys())
        manifests = self.reconciler.reconcile(
            decision=decision,
            pools=self.pools,
            pod=self.pod,
            required_replicas=req_replicas,
            compromised_pools=compromised_list,
        )

        p99_wait_sec = ggc_quantile_wait_seconds(
            arrival_rate_rps=buffered_arrival_rps,
            vcpus=float(decision.total_provisioned_vcpus),
            per_vcpu_mu=float(self.pod.target_rps_per_pod),
            quantile=0.99,
            arrival_cv2=self.policy.arrival_cv2,
            service_cv2=self.policy.service_cv2,
        )

        return K8sControlCycleReport(
            conformal=conformal_res,
            observed_rps=obs,
            design_rps=design_rps,
            rps_ramp_rate_per_sec=ramp_rate,
            required_pod_replicas=req_replicas,
            total_schedulable_pods=int(decision.total_provisioned_vcpus),
            expected_p99_queue_wait_ms=p99_wait_sec * 1000.0,
            portfolio=decision,
            manifests=manifests,
            compromised_pools=compromised_list,
        )

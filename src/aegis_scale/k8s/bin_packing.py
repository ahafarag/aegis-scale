"""Two-dimensional CPU and memory Kubernetes pod bin-packing and G/G/c replica sizing.

Translates heterogeneous EC2 node pools into exact integer pod-slot capacities after
accounting for DaemonSet/kubelet reservations, preventing memory OOM or stranded vCPUs
on skewed workloads, and sizes proactive pod replicas with node boot lead-time buffering.
"""

from __future__ import annotations

import math
from pydantic import BaseModel, Field

from aegis_scale.config import ComputePool
from aegis_scale.math_core.queuing import required_safe_vcpus


class PodResources(BaseModel):
    """Resource requests and steady-state per-pod throughput capacity for a Kubernetes workload."""

    name: str = Field(..., min_length=1, description="Kubernetes Deployment or StatefulSet name")
    cpu_vcpu: float = Field(..., gt=0.0, description="Pod CPU request in vCPUs (cores)")
    memory_gib: float = Field(..., gt=0.0, description="Pod memory request in GiB")
    target_rps_per_pod: float = Field(
        ...,
        gt=0.0,
        description="Mean service rate mu (requests/sec) sustainable by a single pod replica",
    )


class K8sClusterOverhead(BaseModel):
    """Per-node system/DaemonSet resource reservations and EC2-to-Ready boot latency."""

    daemonset_cpu_vcpu: float = Field(
        default=0.25,
        ge=0.0,
        description="Node CPU reserved for kubelet, CNI, and monitoring DaemonSets",
    )
    daemonset_memory_gib: float = Field(
        default=0.75,
        ge=0.0,
        description="Node memory in GiB reserved for kubelet, CNI, and monitoring DaemonSets",
    )
    node_boot_seconds: float = Field(
        default=65.0,
        ge=0.0,
        description="End-to-end Karpenter EC2 launch, kubelet join, and image pull latency in seconds",
    )


def pods_per_node(
    pool: ComputePool,
    pod: PodResources,
    overhead: K8sClusterOverhead = K8sClusterOverhead(),
) -> int:
    """Compute exact 2D integer pod capacity of a single EC2 node after DaemonSet overhead."""
    allocatable_cpu = float(pool.vcpu) - float(overhead.daemonset_cpu_vcpu)
    allocatable_mem = float(pool.memory_gib) - float(overhead.daemonset_memory_gib)
    if allocatable_cpu <= 0.0 or allocatable_mem <= 0.0:
        return 0

    cpu_bound = allocatable_cpu / float(pod.cpu_vcpu)
    mem_bound = allocatable_mem / float(pod.memory_gib)
    return max(0, math.floor(min(cpu_bound, mem_bound)))


def build_2d_effective_pools(
    pools: list[ComputePool],
    pod: PodResources,
    overhead: K8sClusterOverhead = K8sClusterOverhead(),
) -> tuple[list[ComputePool], dict[str, int]]:
    """Project physical EC2 ComputePools into effective pod-slot pools for CVaR MILP optimization.

    Each schedulable pool's effective `vcpu` attribute is set to `max(1, pods_cap)` so the
    MILP measures fleet capacity directly in integer schedulable Pod slots while preserving
    true per-node hourly pricing and correlated Spot hazard metadata. Pools unable to fit
    at least one pod (`pods_cap < 1`) are filtered out.
    """
    effective_pools: list[ComputePool] = []
    pods_per_node_map: dict[str, int] = {}

    for pool in pools:
        pods_cap = pods_per_node(pool, pod, overhead)
        if pods_cap < 1:
            continue
        pods_per_node_map[pool.pool_id] = pods_cap
        effective_pools.append(pool.model_copy(update={"vcpu": max(1, pods_cap)}))

    return effective_pools, pods_per_node_map


def required_safe_pod_replicas(
    design_rps: float,
    pod: PodResources,
    sla_wait_ms: float = 25.0,
    arrival_cv2: float = 1.6,
    service_cv2: float = 1.2,
    cold_start_surge_rate_rps_per_sec: float = 0.0,
    node_boot_seconds: float = 65.0,
) -> int:
    """Compute minimum integer pod replicas guaranteeing P99 queueing delay <= sla_wait_ms.

    Models each pod replica as a server in a G/G/c system with service rate
    `per_vcpu_mu = pod.target_rps_per_pod`, adding a proactive cold-start demand buffer
    `max(0.0, cold_start_surge_rate_rps_per_sec * node_boot_seconds)` prior to ceiling.
    """
    lead_time_buffer_rps = max(
        0.0, float(cold_start_surge_rate_rps_per_sec) * float(node_boot_seconds)
    )
    buffered_arrival_rps = max(0.0, float(design_rps)) + lead_time_buffer_rps

    continuous_pods = required_safe_vcpus(
        arrival_rate_rps=buffered_arrival_rps,
        per_vcpu_mu=float(pod.target_rps_per_pod),
        sla_wait_ms=float(sla_wait_ms),
        quantile=0.99,
        arrival_cv2=float(arrival_cv2),
        service_cv2=float(service_cv2),
        min_vcpus=1.0,
    )
    return max(1, int(math.ceil(continuous_pods)))

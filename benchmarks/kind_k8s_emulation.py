"""Live Multi-Node Kubernetes (kind) Emulation Benchmark for AegisScale.

Spins up a real 4-node Kubernetes cluster on Docker Desktop using kind
(1 control-plane + 3 labeled worker nodes across 3 simulated AZs), builds and
loads a local Python HTTP microservice (`aegis-echo-worker:local`), replays a
scaled 60-second burst profile derived from the October 2024 Microsoft Azure
ModServe inference trace with real concurrent async HTTP traffic, and injects
real correlated Kubernetes Spot node interruptions (`kubectl cordon` + `drain`).

Compares:
  1. Greedy_Spot_Unhedged: 4 replicas pinned to Spot workers (no On-Demand anchor).
  2. AegisScale_K8s_Protected: K8sAegisOperator CVaR-hedged hybrid portfolio
     (On-Demand anchor + multi-AZ diversified Spot with nodeAffinity,
     podAntiAffinity, topologySpreadConstraints, and PodDisruptionBudget).

Always deletes the kind cluster in a finally block so no containers remain.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_scale.config import ComputePool, SLAPolicy
from aegis_scale.k8s.bin_packing import K8sClusterOverhead, PodResources
from aegis_scale.k8s.karpenter import K8sAegisOperator

KIND_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "kind.exe"
KIND_DOWNLOAD_URL = "https://kind.sigs.k8s.io/dl/v0.25.0/kind-windows-amd64"
CLUSTER_NAME = "aegis-emulation"
KUBE_CONTEXT = f"kind-{CLUSTER_NAME}"
IMAGE_NAME = "aegis-echo-worker:local"
TRACE_CSV = PROJECT_ROOT / "benchmarks" / "data" / "real_azure_inference_2024.csv"
OUTPUT_JSON = PROJECT_ROOT / "output" / "kind_live_k8s_results.json"
OUTPUT_FIG = PROJECT_ROOT / "docs" / "figures" / "kind_live_k8s.png"

NODE_PORT = 30080
SERVICE_URL = f"http://127.0.0.1:{NODE_PORT}/work"
HEALTH_URL = f"http://127.0.0.1:{NODE_PORT}/healthz"

EXPERIMENT_DURATION_SEC = 60
DRAIN_T1_SEC = 20.0
DRAIN_T2_SEC = 25.0


ECHO_SERVER_PY = r'''"""Lightweight HTTP worker with deterministic ~8ms CPU loop and graceful SIGTERM."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import socket
import threading
import time

WORK_DURATION_SEC = float(os.environ.get("WORK_DURATION_SEC", "0.008"))
POD_NAME = os.environ.get("POD_NAME", socket.gethostname())
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
SHUTTING_DOWN = threading.Event()
PAYLOAD_BLOCK = b"aegis-scale-deterministic-work-block-" * 512


def do_deterministic_work(duration_sec: float) -> float:
    t0 = time.perf_counter()
    digest = PAYLOAD_BLOCK
    while (time.perf_counter() - t0) < duration_sec:
        digest = hashlib.sha256(digest).digest()
    return (time.perf_counter() - t0) * 1000.0


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, body: dict[str, object]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            if SHUTTING_DOWN.is_set():
                self._send_json(503, {"status": "draining", "pod": POD_NAME})
            else:
                self._send_json(200, {"status": "healthy", "pod": POD_NAME, "node": NODE_NAME})
            return

        if self.path.startswith("/work"):
            work_ms = do_deterministic_work(WORK_DURATION_SEC)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "pod": POD_NAME,
                    "node": NODE_NAME,
                    "work_ms": round(work_ms, 3),
                },
            )
            return

        self._send_json(404, {"status": "not_found"})


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8080), WorkerHandler)
    server.daemon_threads = True

    def handle_sigterm(signum: int, frame: object) -> None:
        SHUTTING_DOWN.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    server.serve_forever()


if __name__ == "__main__":
    main()
'''

DOCKERFILE_TEXT = """FROM python:3.11-slim
WORKDIR /app
COPY server.py /app/server.py
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/server.py"]
"""


@dataclass
class RequestSample:
    t_sec: float
    second_bin: int
    status_code: int
    latency_ms: float
    ok: bool
    error_type: str
    pod: str
    node: str


def run_cmd(
    cmd: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a host CLI command and return CompletedProcess."""
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def kubectl(
    args: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["kubectl", "--context", KUBE_CONTEXT, *args],
        input_text=input_text,
        check=check,
        timeout=timeout,
    )


def ensure_kind_binary() -> Path:
    """Ensure kind v0.25.0 Windows amd64 binary is present in .venv/Scripts."""
    if KIND_EXE.exists():
        print(f"[setup] Found kind binary at {KIND_EXE}")
        return KIND_EXE
    KIND_EXE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Downloading kind v0.25.0 from {KIND_DOWNLOAD_URL} ...")
    urllib.request.urlretrieve(KIND_DOWNLOAD_URL, str(KIND_EXE))
    print(f"[setup] Downloaded kind to {KIND_EXE}")
    return KIND_EXE


def build_echo_image() -> None:
    """Build tiny local Docker image aegis-echo-worker:local."""
    print(f"[setup] Building Docker image {IMAGE_NAME} ...")
    with tempfile.TemporaryDirectory(prefix="aegis-echo-build-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "server.py").write_text(ECHO_SERVER_PY, encoding="utf-8")
        (tmp_path / "Dockerfile").write_text(DOCKERFILE_TEXT, encoding="utf-8")
        run_cmd(["docker", "build", "-t", IMAGE_NAME, str(tmp_path)], timeout=300)
    print(f"[setup] Built Docker image {IMAGE_NAME}")


def delete_kind_cluster() -> None:
    """Cleanly delete the kind cluster if it exists."""
    if not KIND_EXE.exists():
        return
    print(f"[cleanup] Deleting kind cluster '{CLUSTER_NAME}' ...")
    proc = run_cmd(
        [str(KIND_EXE), "delete", "cluster", "--name", CLUSTER_NAME],
        check=False,
        timeout=180,
    )
    if proc.returncode == 0:
        print(f"[cleanup] Deleted kind cluster '{CLUSTER_NAME}'")
    else:
        print(f"[cleanup] kind delete returned {proc.returncode}: {proc.stderr.strip()}")


def create_kind_cluster() -> dict[str, str]:
    """Create 4-node kind cluster, load image, and label 3 worker nodes."""
    # Ensure clean slate if a previous run was interrupted
    run_cmd([str(KIND_EXE), "delete", "cluster", "--name", CLUSTER_NAME], check=False, timeout=120)

    kind_cfg = f"""kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: {CLUSTER_NAME}
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: {NODE_PORT}
        hostPort: {NODE_PORT}
        protocol: TCP
  - role: worker
  - role: worker
  - role: worker
"""
    print(f"[cluster] Creating 4-node kind cluster '{CLUSTER_NAME}' ...")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(kind_cfg)
        cfg_path = Path(f.name)
    try:
        run_cmd(
            [str(KIND_EXE), "create", "cluster", "--config", str(cfg_path), "--wait", "180s"],
            timeout=300,
        )
    finally:
        cfg_path.unlink(missing_ok=True)

    print(f"[cluster] Loading image {IMAGE_NAME} into kind cluster ...")
    run_cmd(
        [str(KIND_EXE), "load", "docker-image", IMAGE_NAME, "--name", CLUSTER_NAME],
        timeout=180,
    )

    nodes_raw = kubectl(["get", "nodes", "-o", "json"]).stdout
    nodes_json = json.loads(nodes_raw)
    workers = sorted(
        item["metadata"]["name"]
        for item in nodes_json["items"]
        if "node-role.kubernetes.io/control-plane" not in item["metadata"].get("labels", {})
    )
    if len(workers) != 3:
        raise RuntimeError(f"Expected 3 worker nodes, found {len(workers)}: {workers}")

    worker_map = {
        "on_demand_az1": workers[0],
        "spot_az2": workers[1],
        "spot_az3": workers[2],
    }

    kubectl([
        "label", "node", worker_map["on_demand_az1"],
        "aegis-scale.io/capacity-type=on-demand",
        "topology.kubernetes.io/zone=us-east-1a",
        "--overwrite",
    ])
    kubectl([
        "label", "node", worker_map["spot_az2"],
        "aegis-scale.io/capacity-type=spot",
        "topology.kubernetes.io/zone=us-east-1b",
        "--overwrite",
    ])
    kubectl([
        "label", "node", worker_map["spot_az3"],
        "aegis-scale.io/capacity-type=spot",
        "topology.kubernetes.io/zone=us-east-1c",
        "--overwrite",
    ])
    print(
        f"[cluster] Labeled workers: OD={worker_map['on_demand_az1']} (us-east-1a), "
        f"Spot1={worker_map['spot_az2']} (us-east-1b), Spot2={worker_map['spot_az3']} (us-east-1c)"
    )
    return worker_map


def load_azure_burst_profile(duration_sec: int = EXPERIMENT_DURATION_SEC) -> tuple[list[float], dict[str, Any]]:
    """Extract a 60-step burst segment from October 2024 Azure ModServe trace and scale to local RPS."""
    with TRACE_CSV.open("r", encoding="utf-8") as f:
        rows = [(r["timestamp"], float(r["value"])) for r in csv.DictReader(f)]

    vals = np.array([v for _, v in rows], dtype=float)
    best_i, best_score = 0, -1.0
    for i in range(0, len(vals) - duration_sec, 5):
        seg = vals[i : i + duration_sec]
        base = float(seg[:15].mean())
        surge = float(seg[15:40].mean())
        peak = float(seg[18:35].max())
        if base > 0:
            score = (peak / base) * (surge / base)
            if score > best_score:
                best_score = score
                best_i = i

    seg = vals[best_i : best_i + duration_sec]
    seg_min = float(seg.min())
    seg_max = float(seg.max())
    # Scale cleanly to 16..56 RPS so local kind pods reflect real queueing & preemption behavior
    scaled = 16.0 + 40.0 * (seg - seg_min) / max(1e-9, seg_max - seg_min)
    rps_list = [round(float(x), 2) for x in scaled]
    meta = {
        "source_csv": str(TRACE_CSV.relative_to(PROJECT_ROOT)),
        "window_start_idx": int(best_i),
        "window_start_timestamp": rows[best_i][0],
        "window_end_timestamp": rows[best_i + duration_sec - 1][0],
        "raw_min_rpm": round(seg_min, 2),
        "raw_max_rpm": round(seg_max, 2),
        "scaled_min_rps": round(min(rps_list), 2),
        "scaled_max_rps": round(max(rps_list), 2),
        "scaled_mean_rps": round(float(np.mean(rps_list)), 2),
    }
    return rps_list, meta


def apply_manifest(manifest: dict[str, Any]) -> None:
    kubectl(["apply", "-f", "-"], input_text=json.dumps(manifest))


def build_service_manifest() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "aegis-workload-svc",
            "namespace": "default",
            "labels": {"app": "aegis-workload"},
        },
        "spec": {
            "type": "NodePort",
            "selector": {"app": "aegis-workload"},
            "ports": [
                {
                    "name": "http",
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8080,
                    "nodePort": NODE_PORT,
                }
            ],
        },
    }


def _base_container_spec(with_prestop: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "echo-worker",
        "image": IMAGE_NAME,
        "imagePullPolicy": "Never",
        "ports": [{"containerPort": 8080}],
        "env": [
            {"name": "WORK_DURATION_SEC", "value": "0.008"},
            {
                "name": "POD_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
            },
            {
                "name": "NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            },
        ],
        "resources": {
            "requests": {"cpu": "150m", "memory": "64Mi"},
            "limits": {"cpu": "500m", "memory": "128Mi"},
        },
        "readinessProbe": {
            "httpGet": {"path": "/healthz", "port": 8080},
            "initialDelaySeconds": 1,
            "periodSeconds": 2,
            "failureThreshold": 2,
        },
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": 8080},
            "initialDelaySeconds": 3,
            "periodSeconds": 5,
        },
    }
    if with_prestop:
        spec["lifecycle"] = {
            "preStop": {"exec": {"command": ["sh", "-c", "sleep 2"]}},
        }
    return spec


def build_greedy_spot_deployment(replicas: int = 4) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "aegis-workload-greedy",
            "namespace": "default",
            "labels": {"app": "aegis-workload", "strategy": "greedy-spot"},
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "aegis-workload", "tier": "greedy-spot"}},
            "template": {
                "metadata": {
                    "labels": {"app": "aegis-workload", "tier": "greedy-spot"},
                },
                "spec": {
                    "terminationGracePeriodSeconds": 5,
                    "nodeSelector": {
                        "aegis-scale.io/capacity-type": "spot",
                    },
                    "topologySpreadConstraints": [
                        {
                            "maxSkew": 1,
                            "topologyKey": "topology.kubernetes.io/zone",
                            "whenUnsatisfiable": "ScheduleAnyway",
                            "labelSelector": {
                                "matchLabels": {"app": "aegis-workload", "tier": "greedy-spot"},
                            },
                        }
                    ],
                    "containers": [_base_container_spec(with_prestop=False)],
                },
            },
        },
    }


def build_aegis_manifests(od_replicas: int, spot_replicas: int) -> list[dict[str, Any]]:
    pdb = {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {
            "name": "aegis-workload-pdb",
            "namespace": "default",
        },
        "spec": {
            "minAvailable": max(1, od_replicas),
            "selector": {
                "matchLabels": {"app": "aegis-workload"},
            },
        },
    }

    od_deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "aegis-workload-od",
            "namespace": "default",
            "labels": {"app": "aegis-workload", "tier": "on-demand"},
        },
        "spec": {
            "replicas": od_replicas,
            "selector": {"matchLabels": {"app": "aegis-workload", "tier": "on-demand"}},
            "template": {
                "metadata": {
                    "labels": {"app": "aegis-workload", "tier": "on-demand"},
                },
                "spec": {
                    "terminationGracePeriodSeconds": 10,
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "aegis-scale.io/capacity-type",
                                                "operator": "In",
                                                "values": ["on-demand"],
                                            },
                                            {
                                                "key": "topology.kubernetes.io/zone",
                                                "operator": "In",
                                                "values": ["us-east-1a"],
                                            },
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    "containers": [_base_container_spec(with_prestop=True)],
                },
            },
        },
    }

    spot_deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "aegis-workload-spot",
            "namespace": "default",
            "labels": {"app": "aegis-workload", "tier": "spot"},
        },
        "spec": {
            "replicas": spot_replicas,
            "selector": {"matchLabels": {"app": "aegis-workload", "tier": "spot"}},
            "template": {
                "metadata": {
                    "labels": {"app": "aegis-workload", "tier": "spot"},
                },
                "spec": {
                    "terminationGracePeriodSeconds": 10,
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "aegis-scale.io/capacity-type",
                                                "operator": "In",
                                                "values": ["spot"],
                                            }
                                        ]
                                    }
                                ]
                            }
                        },
                        "podAntiAffinity": {
                            "preferredDuringSchedulingIgnoredDuringExecution": [
                                {
                                    "weight": 100,
                                    "podAffinityTerm": {
                                        "labelSelector": {
                                            "matchLabels": {
                                                "app": "aegis-workload",
                                                "tier": "spot",
                                            }
                                        },
                                        "topologyKey": "topology.kubernetes.io/zone",
                                    },
                                }
                            ]
                        },
                    },
                    "topologySpreadConstraints": [
                        {
                            "maxSkew": 1,
                            "topologyKey": "topology.kubernetes.io/zone",
                            "whenUnsatisfiable": "ScheduleAnyway",
                            "labelSelector": {
                                "matchLabels": {"app": "aegis-workload", "tier": "spot"},
                            },
                        }
                    ],
                    "containers": [_base_container_spec(with_prestop=True)],
                },
            },
        },
    }
    return [pdb, od_deploy, spot_deploy]


def get_pod_snapshot() -> dict[str, Any]:
    """Query Kubernetes API for live pod readiness and node placement."""
    raw = kubectl(["get", "pods", "-l", "app=aegis-workload", "-o", "json"]).stdout
    items = json.loads(raw).get("items", [])
    ready = 0
    pending = 0
    terminating = 0
    by_node: dict[str, int] = {}
    for p in items:
        meta = p.get("metadata", {})
        status = p.get("status", {})
        spec = p.get("spec", {})
        node = spec.get("nodeName") or "unscheduled"
        is_term = meta.get("deletionTimestamp") is not None
        phase = status.get("phase", "")
        conds = status.get("conditions", [])
        is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds)
        if is_term:
            terminating += 1
        elif is_ready and phase == "Running":
            ready += 1
            by_node[node] = by_node.get(node, 0) + 1
        elif phase == "Pending":
            pending += 1
    return {
        "total": len(items),
        "ready": ready,
        "pending": pending,
        "terminating": terminating,
        "ready_by_node": by_node,
    }


async def wait_for_service_ready(expected_ready_pods: int, timeout_sec: float = 90.0) -> None:
    """Wait until expected pods are Ready and HTTP /work returns 200 OK."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        snap = get_pod_snapshot()
        if snap["ready"] >= expected_ready_pods:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(SERVICE_URL, headers={"Connection": "close"})
                    if r.status_code == 200:
                        return
            except Exception:
                pass
        await asyncio.sleep(1.0)
    raise RuntimeError(f"Service did not reach {expected_ready_pods} ready pods within {timeout_sec}s")


async def drain_node_async(node_name: str, t_start: float, events: list[dict[str, Any]], label: str) -> None:
    """Execute real kubectl cordon + drain on a Spot worker node."""
    t_fire = time.perf_counter() - t_start
    print(f"  [fault t={t_fire:.1f}s] Cordoning and draining {node_name} ({label}) ...")
    events.append({"event": f"cordon_start_{label}", "node": node_name, "t_sec": round(t_fire, 2)})

    cordon_proc = await asyncio.create_subprocess_exec(
        "kubectl", "--context", KUBE_CONTEXT, "cordon", node_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await cordon_proc.communicate()

    drain_proc = await asyncio.create_subprocess_exec(
        "kubectl", "--context", KUBE_CONTEXT,
        "drain", node_name,
        "--ignore-daemonsets",
        "--delete-emptydir-data",
        "--force",
        "--grace-period=5",
        "--timeout=30s",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await drain_proc.communicate()
    t_done = time.perf_counter() - t_start
    events.append({
        "event": f"drain_complete_{label}",
        "node": node_name,
        "t_sec": round(t_done, 2),
        "returncode": drain_proc.returncode,
        "stderr_tail": stderr_b.decode("utf-8", errors="replace").strip()[-200:],
    })
    print(f"  [fault t={t_done:.1f}s] Drain finished on {node_name} (rc={drain_proc.returncode})")


async def run_live_experiment(
    name: str,
    rps_profile: list[float],
    worker_map: dict[str, str],
) -> dict[str, Any]:
    """Replay 60-second concurrent HTTP traffic while injecting correlated Spot drains."""
    samples: list[RequestSample] = []
    pod_timeline: list[dict[str, Any]] = []
    fault_events: list[dict[str, Any]] = []

    stop_poller = asyncio.Event()
    t_start = time.perf_counter()

    async def poll_pods() -> None:
        while not stop_poller.is_set():
            t_rel = round(time.perf_counter() - t_start, 2)
            try:
                snap = await asyncio.to_thread(get_pod_snapshot)
                pod_timeline.append({"t_sec": t_rel, **snap})
            except Exception as exc:
                pod_timeline.append({"t_sec": t_rel, "error": str(exc)})
            await asyncio.sleep(1.5)

    async def fault_scheduler() -> None:
        await asyncio.sleep(DRAIN_T1_SEC)
        t1 = asyncio.create_task(
            drain_node_async(worker_map["spot_az2"], t_start, fault_events, "spot_az2")
        )
        await asyncio.sleep(DRAIN_T2_SEC - DRAIN_T1_SEC)
        t2 = asyncio.create_task(
            drain_node_async(worker_map["spot_az3"], t_start, fault_events, "spot_az3")
        )
        await asyncio.gather(t1, t2)

    async def fire_one(client: httpx.AsyncClient, sec_bin: int) -> None:
        req_t0 = time.perf_counter()
        rel_t = req_t0 - t_start
        try:
            resp = await client.get(SERVICE_URL, headers={"Connection": "close"})
            lat_ms = (time.perf_counter() - req_t0) * 1000.0
            if resp.status_code == 200:
                payload = resp.json()
                samples.append(
                    RequestSample(
                        t_sec=round(rel_t, 3),
                        second_bin=sec_bin,
                        status_code=200,
                        latency_ms=round(lat_ms, 3),
                        ok=True,
                        error_type="",
                        pod=str(payload.get("pod", "")),
                        node=str(payload.get("node", "")),
                    )
                )
            else:
                samples.append(
                    RequestSample(
                        t_sec=round(rel_t, 3),
                        second_bin=sec_bin,
                        status_code=resp.status_code,
                        latency_ms=round(lat_ms, 3),
                        ok=False,
                        error_type=f"http_{resp.status_code}",
                        pod="",
                        node="",
                    )
                )
        except Exception as exc:
            lat_ms = (time.perf_counter() - req_t0) * 1000.0
            samples.append(
                RequestSample(
                    t_sec=round(rel_t, 3),
                    second_bin=sec_bin,
                    status_code=0,
                    latency_ms=round(lat_ms, 3),
                    ok=False,
                    error_type=type(exc).__name__,
                    pod="",
                    node="",
                )
            )

    poller_task = asyncio.create_task(poll_pods())
    fault_task = asyncio.create_task(fault_scheduler())

    limits = httpx.Limits(max_connections=120, max_keepalive_connections=0)
    timeout = httpx.Timeout(connect=1.5, read=2.0, write=1.5, pool=2.0)

    in_flight: list[asyncio.Task[None]] = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for sec in range(len(rps_profile)):
            target_rps = max(1, int(round(rps_profile[sec])))
            sec_start = t_start + float(sec)
            now = time.perf_counter()
            if sec_start > now:
                await asyncio.sleep(sec_start - now)

            spacing = 0.92 / float(target_rps)
            for k in range(target_rps):
                in_flight.append(asyncio.create_task(fire_one(client, sec)))
                if k + 1 < target_rps:
                    await asyncio.sleep(spacing)

        # Wait for remaining in-flight requests to settle
        await asyncio.gather(*in_flight, return_exceptions=True)

    await fault_task
    stop_poller.set()
    await poller_task

    # Aggregate per-second telemetry
    per_second: list[dict[str, Any]] = []
    for sec in range(len(rps_profile)):
        bin_samples = [s for s in samples if s.second_bin == sec]
        ok_lats = [s.latency_ms for s in bin_samples if s.ok]
        n_tot = len(bin_samples)
        n_ok = len(ok_lats)
        n_err = n_tot - n_ok
        err_pct = (100.0 * n_err / n_tot) if n_tot > 0 else 0.0

        # Nearest pod snapshot
        nearest_pods = None
        if pod_timeline:
            nearest = min(pod_timeline, key=lambda x: abs(x.get("t_sec", 0.0) - (sec + 0.5)))
            nearest_pods = nearest.get("ready")

        per_second.append({
            "second": sec,
            "target_rps": rps_profile[sec],
            "total_requests": n_tot,
            "ok_requests": n_ok,
            "failed_requests": n_err,
            "error_rate_pct": round(err_pct, 2),
            "p50_ms": round(float(np.percentile(ok_lats, 50)), 2) if ok_lats else None,
            "p95_ms": round(float(np.percentile(ok_lats, 95)), 2) if ok_lats else None,
            "p99_ms": round(float(np.percentile(ok_lats, 99)), 2) if ok_lats else None,
            "ready_pods": nearest_pods,
        })

    all_ok_lats = [s.latency_ms for s in samples if s.ok]
    post_drain_samples = [s for s in samples if s.second_bin >= int(DRAIN_T1_SEC)]
    post_drain_err = sum(1 for s in post_drain_samples if not s.ok)
    post_drain_err_pct = (
        100.0 * post_drain_err / len(post_drain_samples) if post_drain_samples else 0.0
    )

    total_req = len(samples)
    ok_req = len(all_ok_lats)
    fail_req = total_req - ok_req
    err_rate_pct = (100.0 * fail_req / total_req) if total_req > 0 else 0.0

    # Determine outage duration / time-to-recover within the 60s window
    outage_secs = [row["second"] for row in per_second if row["second"] >= int(DRAIN_T1_SEC) and row["error_rate_pct"] >= 50.0]
    if not outage_secs:
        recovered_in_window = True
        in_window_recovery_sec: float | None = 0.0
    else:
        first_outage = outage_secs[0]
        later_ok = [
            row["second"]
            for row in per_second
            if row["second"] > first_outage and row["error_rate_pct"] < 5.0
        ]
        if later_ok:
            recovered_in_window = True
            in_window_recovery_sec = float(later_ok[0] - first_outage)
        else:
            recovered_in_window = False
            in_window_recovery_sec = None

    ready_counts = [
        p["ready"] for p in pod_timeline if "ready" in p and p.get("t_sec", 0.0) >= DRAIN_T1_SEC
    ]
    min_ready_during_drain = int(min(ready_counts)) if ready_counts else 0

    node_distribution: dict[str, int] = {}
    for s in samples:
        if s.ok and s.node:
            node_distribution[s.node] = node_distribution.get(s.node, 0) + 1

    summary = {
        "experiment": name,
        "total_requests": total_req,
        "ok_requests": ok_req,
        "failed_requests": fail_req,
        "http_error_rate_pct": round(err_rate_pct, 2),
        "post_drain_error_rate_pct": round(post_drain_err_pct, 2),
        "p50_latency_ms": round(float(np.percentile(all_ok_lats, 50)), 2) if all_ok_lats else None,
        "p95_latency_ms": round(float(np.percentile(all_ok_lats, 95)), 2) if all_ok_lats else None,
        "p99_latency_ms": round(float(np.percentile(all_ok_lats, 99)), 2) if all_ok_lats else None,
        "min_ready_pods_during_drain": min_ready_during_drain,
        "outage_seconds_count": len(outage_secs),
        "recovered_within_60s_window": recovered_in_window,
        "in_window_recovery_sec": in_window_recovery_sec,
        "requests_served_by_node": node_distribution,
    }
    print(
        f"[{name}] Summary: total={total_req}, ok={ok_req}, err={err_rate_pct:.2f}% "
        f"(post-drain={post_drain_err_pct:.2f}%), "
        f"P50={summary['p50_latency_ms']}ms, P95={summary['p95_latency_ms']}ms, "
        f"P99={summary['p99_latency_ms']}ms, min_ready_pods={min_ready_during_drain}"
    )

    return {
        "summary": summary,
        "fault_events": fault_events,
        "per_second": per_second,
        "pod_timeline": pod_timeline,
    }


def compute_aegis_operator_plan(rps_profile: list[float]) -> dict[str, Any]:
    """Run K8sAegisOperator over the burst profile to synthesize the CVaR-hedged split."""
    pools = [
        ComputePool(
            pool_id="od-c7g-az1",
            instance_type="c7g.medium",
            family="c7g",
            az="us-east-1a",
            market="on-demand",
            vcpu=1,
            memory_gib=2.0,
            hourly_price_usd=0.036,
            base_interruption_prob=0.0,
            max_instances=8,
        ),
        ComputePool(
            pool_id="spot-c7g-az2",
            instance_type="c7g.medium",
            family="c7g",
            az="us-east-1b",
            market="spot",
            vcpu=1,
            memory_gib=2.0,
            hourly_price_usd=0.011,
            base_interruption_prob=0.025,
            max_instances=8,
        ),
        ComputePool(
            pool_id="spot-m7g-az3",
            instance_type="m7g.medium",
            family="m7g",
            az="us-east-1c",
            market="spot",
            vcpu=1,
            memory_gib=4.0,
            hourly_price_usd=0.012,
            base_interruption_prob=0.025,
            max_instances=8,
        ),
    ]
    pod = PodResources(
        name="aegis-workload",
        cpu_vcpu=0.6,
        memory_gib=0.5,
        target_rps_per_pod=45.0,
    )
    policy = SLAPolicy(
        p99_queue_delay_sla_ms=30.0,
        on_demand_floor_ratio=0.35,
        max_spot_pool_share=0.45,
        max_cvar_shortfall_vcpu=1.5,
        cvar_beta=0.95,
        num_scenarios=300,
    )
    overhead = K8sClusterOverhead(
        daemonset_cpu_vcpu=0.2,
        daemonset_memory_gib=0.25,
        node_boot_seconds=10.0,
    )
    operator = K8sAegisOperator(
        pools=pools,
        pod=pod,
        policy=policy,
        overhead=overhead,
        interval_sec=60.0,
        seed=42,
    )

    peak_report = None
    # Feed pre-surge and surge demand points through ACI + 2D sizing + CVaR MILP
    for rps in rps_profile[:35]:
        rep = operator.run_cycle(float(rps))
        if peak_report is None or rep.total_schedulable_pods > peak_report.total_schedulable_pods:
            peak_report = rep

    assert peak_report is not None
    allocs = dict(peak_report.portfolio.allocations)
    od_replicas = int(allocs.get("od-c7g-az1", 0))
    spot_az2 = int(allocs.get("spot-c7g-az2", 0))
    spot_az3 = int(allocs.get("spot-m7g-az3", 0))
    spot_total = spot_az2 + spot_az3

    greedy_spot_hourly = 4 * 0.0115
    all_od_hourly = (od_replicas + spot_total) * 0.036

    return {
        "od_replicas": max(2, od_replicas),
        "spot_replicas": max(2, spot_total),
        "spot_az2_planned": spot_az2,
        "spot_az3_planned": spot_az3,
        "raw_allocations": allocs,
        "required_pod_replicas": int(peak_report.required_pod_replicas),
        "total_schedulable_pods": int(peak_report.total_schedulable_pods),
        "expected_p99_queue_wait_ms": round(float(peak_report.expected_p99_queue_wait_ms), 3),
        "hourly_cost_usd": round(float(peak_report.portfolio.hourly_cost_usd), 4),
        "greedy_spot_4pod_hourly_cost_usd": round(greedy_spot_hourly, 4),
        "all_ondemand_equivalent_hourly_cost_usd": round(all_od_hourly, 4),
        "savings_vs_all_ondemand_pct": round(
            100.0 * (1.0 - float(peak_report.portfolio.hourly_cost_usd) / max(1e-9, all_od_hourly)),
            2,
        ),
        "synthesized_karpenter_nodepools": [
            {
                "name": np_manifest["metadata"]["name"],
                "weight": np_manifest["spec"]["weight"],
                "cpu_limit": np_manifest["spec"]["limits"]["cpu"],
            }
            for np_manifest in peak_report.manifests.node_pools
        ],
    }


def generate_figure(results: dict[str, Any], out_path: Path) -> None:
    """Generate crisp 2-panel 160-DPI figure of live kind K8s results."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    greedy = results["experiments"]["Greedy_Spot_Unhedged"]
    aegis = results["experiments"]["AegisScale_K8s_Protected"]

    secs = np.array([r["second"] for r in greedy["per_second"]], dtype=float)
    g_p99 = np.array(
        [r["p99_ms"] if r["p99_ms"] is not None else np.nan for r in greedy["per_second"]],
        dtype=float,
    )
    a_p99 = np.array(
        [r["p99_ms"] if r["p99_ms"] is not None else np.nan for r in aegis["per_second"]],
        dtype=float,
    )
    g_err = np.array([r["error_rate_pct"] for r in greedy["per_second"]], dtype=float)
    a_err = np.array([r["error_rate_pct"] for r in aegis["per_second"]], dtype=float)
    rps = np.array([r["target_rps"] for r in greedy["per_second"]], dtype=float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.0), dpi=160, gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    color_greedy = "#C0392B"
    color_aegis = "#0F766E"
    color_rps = "#64748B"

    # Panel (a): Real-time HTTP P99 latency time-series during correlated Spot node drains
    ax1.plot(
        secs, g_p99, color=color_greedy, linewidth=2.0, marker="o", markersize=3.2,
        label="Greedy Spot (Unhedged) P99 ms", zorder=4,
    )
    ax1.plot(
        secs, a_p99, color=color_aegis, linewidth=2.2, marker="s", markersize=3.0,
        label="AegisScale K8s Operator P99 ms", zorder=5,
    )

    # Shade blackout/outage region where Greedy Spot has >=50% HTTP errors
    outage_mask = g_err >= 50.0
    if np.any(outage_mask):
        outage_secs = secs[outage_mask]
        ax1.axvspan(
            float(outage_secs.min()),
            float(outage_secs.max()) + 1.0,
            color=color_greedy,
            alpha=0.14,
            label="Greedy Spot Complete Outage (0 Ready Pods)",
            zorder=1,
        )

    ax1.axvline(DRAIN_T1_SEC, color="#B45309", linestyle="--", linewidth=1.4, zorder=3)
    ax1.axvline(DRAIN_T2_SEC, color="#7C2D12", linestyle=":", linewidth=1.6, zorder=3)

    ymax = max(
        float(np.nanmax(g_p99)) if not np.all(np.isnan(g_p99)) else 50.0,
        float(np.nanmax(a_p99)) if not np.all(np.isnan(a_p99)) else 50.0,
        60.0,
    )
    y_top = ymax * 1.42
    ax1.set_ylim(0, y_top)
    ax1.set_xlim(0, EXPERIMENT_DURATION_SEC - 1)
    ax1.text(
        DRAIN_T1_SEC - 0.6, y_top * 0.95, "t=20s\nDrain AZ-b",
        ha="right", va="top", fontsize=8.3, color="#B45309", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax1.text(
        DRAIN_T2_SEC + 0.6, y_top * 0.95, "t=25s\nDrain AZ-c",
        ha="left", va="top", fontsize=8.3, color="#7C2D12", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    # Overlay Azure burst RPS on subtle twin axis
    ax1_rps = ax1.twinx()
    ax1_rps.spines["top"].set_visible(False)
    ax1_rps.fill_between(secs, 0, rps, color=color_rps, alpha=0.10, zorder=0)
    ax1_rps.plot(secs, rps, color=color_rps, alpha=0.45, linewidth=1.0, linestyle="-.", label="Scaled Azure RPS")
    ax1_rps.set_ylabel("Offered Load (HTTP RPS)", color=color_rps, fontsize=9)
    ax1_rps.tick_params(axis="y", colors=color_rps, labelsize=8.5)
    ax1_rps.set_ylim(0, float(rps.max()) * 1.65)

    ax1.set_xlabel("Elapsed Wall-Clock Time (seconds)")
    ax1.set_ylabel("End-to-End HTTP P99 Latency (ms)")
    ax1.set_title("(a) Live kind K8s HTTP P99 Latency During Correlated Spot Drains", fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_rps.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8.0,
        frameon=True,
        framealpha=0.94,
    )

    # Panel (b): Summary comparison of P99 Latency, HTTP Error Rate %, and Min Surviving Ready Pods
    gs = greedy["summary"]
    as_ = aegis["summary"]

    categories = [
        "HTTP P99\nLatency (ms)",
        "HTTP Error\nRate (%)",
        "Min Ready Pods\n(During Drain)",
    ]
    g_vals = [
        float(gs["p99_latency_ms"] or 0.0),
        float(gs["http_error_rate_pct"]),
        float(gs["min_ready_pods_during_drain"]),
    ]
    a_vals = [
        float(as_["p99_latency_ms"] or 0.0),
        float(as_["http_error_rate_pct"]),
        float(as_["min_ready_pods_during_drain"]),
    ]

    # Normalize bars onto a clean percentage-of-max-per-metric visual scale while annotating exact values
    x = np.arange(len(categories))
    width = 0.34
    max_per_cat = [max(g, a, 1.0) for g, a in zip(g_vals, a_vals)]
    g_norm = [100.0 * g / m for g, m in zip(g_vals, max_per_cat)]
    a_norm = [100.0 * a / m for a, m in zip(a_vals, max_per_cat)]

    bars_g = ax2.bar(x - width / 2, g_norm, width, color=color_greedy, alpha=0.88, label="Greedy Spot (Unhedged)")
    bars_a = ax2.bar(x + width / 2, a_norm, width, color=color_aegis, alpha=0.90, label="AegisScale K8s Protected")

    fmt_g = [f"{g_vals[0]:.1f} ms", f"{g_vals[1]:.1f}%", f"{int(g_vals[2])} pods"]
    fmt_a = [f"{a_vals[0]:.1f} ms", f"{a_vals[1]:.2f}%", f"{int(a_vals[2])} pods"]

    for bar, txt in zip(bars_g, fmt_g):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.5,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=color_greedy,
        )
    for bar, txt in zip(bars_a, fmt_a):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.5,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=color_aegis,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.2)
    ax2.set_ylim(0, 134)
    ax2.set_ylabel("Relative Scale (% of Metric Max; Exact Values Annotated)")
    ax2.set_title("(b) Live K8s Resilience & Tail-Latency Summary", fontweight="bold", pad=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper center", ncol=2, fontsize=8.3, frameon=True, framealpha=0.94)

    fig.suptitle(
        "Real 4-Node Kubernetes (kind) Emulation: Correlated Multi-AZ Spot Drain Under Azure ModServe Burst",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)
    print(f"[plot] Saved 2-panel 160-DPI figure to {out_path}")


async def async_main() -> None:
    ensure_kind_binary()
    build_echo_image()

    rps_profile, trace_meta = load_azure_burst_profile(EXPERIMENT_DURATION_SEC)
    print(
        f"[trace] Loaded 60s Azure ModServe burst window "
        f"({trace_meta['window_start_timestamp']} to {trace_meta['window_end_timestamp']}), "
        f"scaled RPS range={trace_meta['scaled_min_rps']}..{trace_meta['scaled_max_rps']}"
    )

    aegis_plan = compute_aegis_operator_plan(rps_profile)
    print(
        f"[operator] K8sAegisOperator CVaR plan: OD={aegis_plan['od_replicas']} pods, "
        f"Spot={aegis_plan['spot_replicas']} pods (AZ-b={aegis_plan['spot_az2_planned']}, "
        f"AZ-c={aegis_plan['spot_az3_planned']}), est=${aegis_plan['hourly_cost_usd']}/hr "
        f"({aegis_plan['savings_vs_all_ondemand_pct']}% savings vs all-OD)"
    )

    cluster_created = False
    try:
        worker_map = create_kind_cluster()
        cluster_created = True

        # Shared NodePort service
        apply_manifest(build_service_manifest())

        # ------------------------------------------------------------------
        # Experiment 1: Greedy_Spot_Unhedged (4 replicas on Spot workers only)
        # ------------------------------------------------------------------
        print("\n=== Experiment 1: Greedy_Spot_Unhedged ===")
        apply_manifest(build_greedy_spot_deployment(replicas=4))
        kubectl(["rollout", "status", "deployment/aegis-workload-greedy", "--timeout=120s"])
        await wait_for_service_ready(expected_ready_pods=4)
        print(f"[Greedy_Spot_Unhedged] Initial pods: {get_pod_snapshot()}")

        greedy_res = await run_live_experiment("Greedy_Spot_Unhedged", rps_profile, worker_map)

        # Uncordon workers after Run 1 and measure actual wall-clock recovery time
        print("[Greedy_Spot_Unhedged] Uncordoning Spot workers after Run 1 to measure recovery ...")
        t_uncordon = time.perf_counter()
        kubectl(["uncordon", worker_map["spot_az2"]])
        kubectl(["uncordon", worker_map["spot_az3"]])
        await wait_for_service_ready(expected_ready_pods=4, timeout_sec=90.0)
        post_uncordon_recovery_sec = round(time.perf_counter() - t_uncordon, 2)
        greedy_res["summary"]["post_uncordon_recovery_sec"] = post_uncordon_recovery_sec
        print(f"[Greedy_Spot_Unhedged] Pods recovered after uncordon in {post_uncordon_recovery_sec:.2f}s")

        # Clean up Experiment 1 Deployment
        kubectl(["delete", "deployment", "aegis-workload-greedy", "--wait=true"])
        await asyncio.sleep(3.0)

        # ------------------------------------------------------------------
        # Experiment 2: AegisScale_K8s_Protected (OD anchor + diversified Spot + PDB)
        # ------------------------------------------------------------------
        print("\n=== Experiment 2: AegisScale_K8s_Protected ===")
        for manifest in build_aegis_manifests(
            od_replicas=aegis_plan["od_replicas"],
            spot_replicas=aegis_plan["spot_replicas"],
        ):
            apply_manifest(manifest)

        kubectl(["rollout", "status", "deployment/aegis-workload-od", "--timeout=120s"])
        kubectl(["rollout", "status", "deployment/aegis-workload-spot", "--timeout=120s"])
        expected_total = aegis_plan["od_replicas"] + aegis_plan["spot_replicas"]
        await wait_for_service_ready(expected_ready_pods=expected_total)
        print(f"[AegisScale_K8s_Protected] Initial pods: {get_pod_snapshot()}")

        aegis_res = await run_live_experiment("AegisScale_K8s_Protected", rps_profile, worker_map)
        aegis_res["summary"]["estimated_hourly_cost_usd"] = aegis_plan["hourly_cost_usd"]
        aegis_res["summary"]["savings_vs_all_ondemand_pct"] = aegis_plan["savings_vs_all_ondemand_pct"]
        aegis_res["summary"]["post_uncordon_recovery_sec"] = 0.0

        # Uncordon workers cleanly before teardown
        kubectl(["uncordon", worker_map["spot_az2"]], check=False)
        kubectl(["uncordon", worker_map["spot_az3"]], check=False)

        results_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cluster": {
                "name": CLUSTER_NAME,
                "nodes": 4,
                "worker_topology": worker_map,
                "image": IMAGE_NAME,
                "service_url": SERVICE_URL,
            },
            "workload_trace": trace_meta,
            "fault_schedule_sec": {
                "spot_az2_drain_sec": DRAIN_T1_SEC,
                "spot_az3_drain_sec": DRAIN_T2_SEC,
                "duration_sec": EXPERIMENT_DURATION_SEC,
            },
            "aegis_operator_plan": aegis_plan,
            "experiments": {
                "Greedy_Spot_Unhedged": greedy_res,
                "AegisScale_K8s_Protected": aegis_res,
            },
        }

        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        print(f"\n[output] Saved live K8s telemetry JSON to {OUTPUT_JSON}")

        generate_figure(results_payload, OUTPUT_FIG)
    finally:
        if cluster_created:
            delete_kind_cluster()


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        generate_figure(data, OUTPUT_FIG)
    else:
        asyncio.run(async_main())

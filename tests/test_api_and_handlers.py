"""Integration tests for FastAPI Control Plane and AWS Lambda event handlers."""

from __future__ import annotations

from typing import Any
from fastapi.testclient import TestClient

from aegis_scale.api.app import create_app
from aegis_scale.cloud.lambda_handler import handle_control_event
from aegis_scale.engine import ControlCycleReport


class _InMemoryStateStore:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None
        self.audits: list[dict[str, Any]] = []

    def load_engine_state(self, cluster_id: str) -> dict[str, Any] | None:
        return self.state

    def save_cycle_state_and_audit(
        self,
        cluster_id: str,
        predictor_state: dict[str, Any],
        report: ControlCycleReport,
    ) -> None:
        self.state = {
            "predictor_state": predictor_state,
            "allocations": dict(report.portfolio.allocations),
        }
        self.audits.append({"step": report.conformal.step, "allocations": dict(report.portfolio.allocations)})


class _StubReconciler:
    def apply_fleet_decision(self, report: ControlCycleReport, previous_total_vcpus: int | None = None) -> dict[str, Any]:
        return {"applied": False, "dry_run": True, "vcpus": report.portfolio.total_provisioned_vcpus}

    @staticmethod
    def build_emf_log_record(cluster_id: str, report: ControlCycleReport) -> str:
        return "{}"


def test_fastapi_endpoints_and_spot_evacuation() -> None:
    client = TestClient(create_app())

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_pools = client.get("/pools")
    assert r_pools.status_code == 200
    assert len(r_pools.json()) >= 15

    r_opt = client.post("/optimize", json={"observed_rps": 3500.0})
    assert r_opt.status_code == 200
    body = r_opt.json()
    assert body["provisioned_vcpus"] >= body["required_safe_vcpus"]
    assert body["cvar_shortfall_vcpu"] <= 1e-6
    assert body["estimated_savings_pct"] > 20.0

    # Trigger 2-minute AWS Spot Interruption Warning on c6a.xlarge in us-east-1a
    r_warn = client.post(
        "/preemption-warning",
        json={
            "instance_type": "c6a.xlarge",
            "az": "us-east-1a",
            "observed_rps": 3500.0,
            "event_kind": "interruption",
        },
    )
    assert r_warn.status_code == 200
    warn_body = r_warn.json()
    assert "spot-c6a.xlarge-us-east-1a" in warn_body["flagged_pools"]
    # Compromised pool with 99% interruption hazard should be evacuated (0 instances)
    assert warn_body["rebalanced_allocations"].get("spot-c6a.xlarge-us-east-1a", 0) == 0

    r_roi = client.post(
        "/simulate-roi",
        json={
            "base_rps": 2500.0,
            "peak_multiplier": 2.4,
            "steps": 48,
            "spot_storm_intensity": 0.8,
            "on_demand_floor_ratio": 0.25,
        },
    )
    assert r_roi.status_code == 200
    roi = r_roi.json()["summary"]
    assert roi["aegis_scale"]["savings_vs_od_pct"] > 20.0
    assert roi["aegis_scale"]["sla_continuity_pct"] >= roi["greedy_spot"]["sla_continuity_pct"]

    r_reg = client.get("/regions")
    assert r_reg.status_code == 200
    assert "eu-west-1" in r_reg.json()["supported_regions"]

    r_k8s = client.post(
        "/k8s/reconcile",
        json={
            "region": "eu-west-1",
            "observed_rps": 3600.0,
            "pod": {
                "name": "inference-worker",
                "cpu_vcpu": 1.0,
                "memory_gib": 6.5,
                "target_rps_per_pod": 90.0,
            },
        },
    )
    assert r_k8s.status_code == 200
    k8s_body = r_k8s.json()
    assert k8s_body["total_schedulable_pods"] >= k8s_body["required_pod_replicas"]
    assert k8s_body["manifests"]["ec2_node_class"]["apiVersion"] == "karpenter.k8s.aws/v1"
    assert len(k8s_body["manifests"]["node_pools"]) >= 2

    # Security hardening tests for public internet exposure (AEGIS_PUBLIC_DEMO=true)
    pub_client = TestClient(create_app(public_demo=True))
    r_pub = pub_client.get("/")
    assert r_pub.status_code == 200
    assert r_pub.headers["X-Frame-Options"] == "DENY"
    assert r_pub.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in r_pub.headers["Content-Security-Policy"]
    # Swagger UI disabled in public demo mode to reduce attack surface
    assert pub_client.get("/docs").status_code == 404
    # Oversized payload (>16 KiB) rejected with HTTP 413
    r_big = pub_client.post("/optimize", content=b"x" * 20_000, headers={"Content-Type": "application/json"})
    assert r_big.status_code == 413
    # Public demo isolates per-request state so Visitor A cannot poison Visitor B
    pub_client.post(
        "/preemption-warning",
        json={"instance_type": "c6a.xlarge", "az": "us-east-1a", "observed_rps": 3000.0},
    )
    r_clean = pub_client.get("/health")
    assert r_clean.json()["aci_steps_processed"] == 0


def test_lambda_handler_state_persistence_and_interruption_event() -> None:
    store = _InMemoryStateStore()
    rec = _StubReconciler()

    # Step 1: Scheduled tick
    res1 = handle_control_event({"observed_rps": 2800.0}, state_store=store, reconciler=rec)
    assert res1["statusCode"] == 200
    assert res1["trigger"] == "ScheduledTick"
    assert len(store.audits) == 1

    # Step 2: AWS EC2 Spot Interruption Warning event
    spot_event = {
        "source": "aws.ec2",
        "detail-type": "EC2 Spot Instance Interruption Warning",
        "detail": {
            "instance-id": "i-0123456789abcdef0",
            "instance-action": "terminate",
            "instance-type": "c6a.xlarge",
            "availability-zone": "us-east-1b",
            "observed_rps": 3000.0,
        },
    }
    res2 = handle_control_event(spot_event, state_store=store, reconciler=rec)
    assert res2["statusCode"] == 200
    assert "spot-c6a.xlarge-us-east-1b" in res2["compromised_pools"]
    assert res2["allocations"].get("spot-c6a.xlarge-us-east-1b", 0) == 0
    assert len(store.audits) == 2

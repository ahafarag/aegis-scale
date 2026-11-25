"""Production AWS boto3 adapters for EC2 Spot pricing, ASG reconciliation, and DynamoDB state."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config

from aegis_scale.config import ComputePool
from aegis_scale.engine import ControlCycleReport

logger = logging.getLogger(__name__)

DEFAULT_BOTO_CONFIG = Config(
    retries={"total_max_attempts": 3, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=10,
    max_pool_connections=25,
)


def _to_decimal_tree(obj: Any) -> Any:
    """Recursively convert floats in nested dicts/lists to Decimal for DynamoDB precision."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 8)))
    if isinstance(obj, dict):
        return {k: _to_decimal_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal_tree(v) for v in obj]
    return obj


def _from_decimal_tree(obj: Any) -> Any:
    """Convert DynamoDB Decimals back to native int/float for mathematical routines."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal_tree(v) for v in obj]
    return obj


class DynamoDBStateStore:
    """Persists online ACI predictor state and FinOps audit trail using DynamoDB Resource API."""

    def __init__(
        self,
        table_name: str,
        session: boto3.Session | None = None,
        region_name: str | None = None,
    ) -> None:
        sess = session or boto3.Session(region_name=region_name)
        self._dynamodb = sess.resource("dynamodb", config=DEFAULT_BOTO_CONFIG)
        self._table = self._dynamodb.Table(table_name)

    def load_engine_state(self, cluster_id: str) -> dict[str, Any] | None:
        """Fetch latest persisted ACI state and active allocations for cluster_id."""
        resp = self._table.get_item(
            Key={"pk": f"CLUSTER#{cluster_id}", "sk": "STATE#CURRENT"}
        )
        item = resp.get("Item")
        if item is None:
            return None
        return _from_decimal_tree(item)

    def save_cycle_state_and_audit(
        self,
        cluster_id: str,
        predictor_state: dict[str, Any],
        report: ControlCycleReport,
    ) -> None:
        """Atomically persist updated ACI state and append an immutable FinOps audit entry."""
        now_iso = datetime.now(timezone.utc).isoformat()
        state_item = _to_decimal_tree(
            {
                "pk": f"CLUSTER#{cluster_id}",
                "sk": "STATE#CURRENT",
                "updated_at": now_iso,
                "predictor_state": predictor_state,
                "allocations": report.portfolio.allocations,
            }
        )
        audit_item = _to_decimal_tree(
            {
                "pk": f"CLUSTER#{cluster_id}",
                "sk": f"AUDIT#{now_iso}",
                "step": report.conformal.step,
                "observed_rps": report.conformal.observed or 0.0,
                "upper_bound_rps": report.conformal.upper_bound,
                "required_vcpus": report.required_vcpus,
                "provisioned_vcpus": report.portfolio.total_provisioned_vcpus,
                "on_demand_vcpus": report.portfolio.on_demand_vcpus,
                "spot_vcpus": report.portfolio.spot_vcpus,
                "hourly_cost_usd": report.portfolio.hourly_cost_usd,
                "baseline_od_cost_usd": report.portfolio.all_on_demand_baseline_cost_usd,
                "estimated_savings_pct": report.portfolio.estimated_savings_pct,
                "cvar_shortfall_vcpu": report.portfolio.cvar_shortfall_vcpu,
                "expected_p99_wait_ms": report.expected_p99_queue_wait_ms,
                "solve_time_ms": report.portfolio.solve_time_ms,
                "allocations": report.portfolio.allocations,
            }
        )
        with self._table.batch_writer() as batch:
            batch.put_item(Item=state_item)
            batch.put_item(Item=audit_item)

    def query_recent_audits(self, cluster_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve latest FinOps audit records in descending chronological order."""
        resp = self._table.query(
            KeyConditionExpression=Key("pk").eq(f"CLUSTER#{cluster_id}")
            & Key("sk").begins_with("AUDIT#"),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_from_decimal_tree(item) for item in resp.get("Items", [])]


class AWSFleetReconciler:
    """Queries live EC2 Spot prices via paginator and reconciles Auto Scaling Group targets."""

    def __init__(
        self,
        asg_name: str,
        dry_run: bool = True,
        max_step_scale_delta_vcpu: int = 256,
        session: boto3.Session | None = None,
        region_name: str | None = None,
    ) -> None:
        sess = session or boto3.Session(region_name=region_name)
        self.asg_name = asg_name
        self.dry_run = dry_run
        self.max_step_scale_delta_vcpu = max_step_scale_delta_vcpu
        self._ec2 = sess.client("ec2", config=DEFAULT_BOTO_CONFIG)
        self._autoscaling = sess.client("autoscaling", config=DEFAULT_BOTO_CONFIG)

    def refresh_spot_prices(self, pools: list[ComputePool]) -> list[ComputePool]:
        """Fetch latest Spot prices using EC2 paginator and return updated ComputePool list."""
        spot_types = sorted({p.instance_type for p in pools if p.market == "spot"})
        if not spot_types:
            return list(pools)

        latest_prices: dict[tuple[str, str], tuple[datetime, float]] = {}
        paginator = self._ec2.get_paginator("describe_spot_price_history")
        page_iter = paginator.paginate(
            InstanceTypes=spot_types,
            ProductDescriptions=["Linux/UNIX"],
        )
        for page in page_iter:
            for entry in page.get("SpotPriceHistory", []):
                key = (entry["InstanceType"], entry["AvailabilityZone"])
                ts = entry["Timestamp"]
                price = float(entry["SpotPrice"])
                prev = latest_prices.get(key)
                if prev is None or ts > prev[0]:
                    latest_prices[key] = (ts, price)

        updated: list[ComputePool] = []
        for pool in pools:
            if pool.market == "spot":
                match = latest_prices.get((pool.instance_type, pool.az))
                if match is not None and match[1] > 0.0:
                    updated.append(pool.model_copy(update={"hourly_price_usd": match[1]}))
                    continue
            updated.append(pool)
        return updated

    def apply_fleet_decision(
        self,
        report: ControlCycleReport,
        previous_total_vcpus: int | None = None,
    ) -> dict[str, Any]:
        """Reconcile target ASG capacity and MixedInstances distribution with circuit-breaker guardrails."""
        decision = report.portfolio
        target_vcpus = decision.total_provisioned_vcpus

        # Safety circuit breaker against runaway single-step scaling jumps
        if previous_total_vcpus is not None:
            delta = target_vcpus - previous_total_vcpus
            if abs(delta) > self.max_step_scale_delta_vcpu:
                capped = previous_total_vcpus + (
                    self.max_step_scale_delta_vcpu if delta > 0 else -self.max_step_scale_delta_vcpu
                )
                logger.warning(
                    "Circuit breaker clamped vCPU change from %d to %d (requested %d)",
                    previous_total_vcpus,
                    capped,
                    target_vcpus,
                )
                target_vcpus = max(4, capped)

        od_base_vcpus = decision.on_demand_vcpus
        above_base_od_pct = (
            100 if decision.spot_vcpus == 0 else 0
        )

        payload = {
            "AutoScalingGroupName": self.asg_name,
            "DesiredCapacity": target_vcpus,
            "MixedInstancesPolicy": {
                "InstancesDistribution": {
                    "OnDemandBaseCapacity": od_base_vcpus,
                    "OnDemandPercentageAboveBaseCapacity": above_base_od_pct,
                    "SpotAllocationStrategy": "price-capacity-optimized",
                }
            },
            "CapacityRebalance": True,
        }

        if self.dry_run:
            return {"applied": False, "dry_run": True, "asg_update_payload": payload}

        self._autoscaling.update_auto_scaling_group(**payload)
        return {"applied": True, "dry_run": False, "asg_update_payload": payload}

    @staticmethod
    def build_emf_log_record(cluster_id: str, report: ControlCycleReport) -> str:
        """Format CloudWatch Embedded Metric Format (EMF) JSON string for zero-latency metrics."""
        emf = {
            "_aws": {
                "Timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "AegisScale/FinOps",
                        "Dimensions": [["ClusterId"]],
                        "Metrics": [
                            {"Name": "HourlyCostUSD", "Unit": "None"},
                            {"Name": "EstimatedSavingsPct", "Unit": "Percent"},
                            {"Name": "CVaRShortfallVCPU", "Unit": "Count"},
                            {"Name": "ExpectedP99QueueWaitMs", "Unit": "Milliseconds"},
                            {"Name": "SolveTimeMs", "Unit": "Milliseconds"},
                        ],
                    }
                ],
            },
            "ClusterId": cluster_id,
            "HourlyCostUSD": round(report.portfolio.hourly_cost_usd, 5),
            "EstimatedSavingsPct": round(report.portfolio.estimated_savings_pct, 2),
            "CVaRShortfallVCPU": round(report.portfolio.cvar_shortfall_vcpu, 4),
            "ExpectedP99QueueWaitMs": round(report.expected_p99_queue_wait_ms, 3),
            "SolveTimeMs": round(report.portfolio.solve_time_ms, 3),
        }
        return json.dumps(emf)

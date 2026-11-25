"""AWS Lambda entrypoints for EventBridge Scheduled Rebalance and EC2 Spot Interruption Warnings."""

from __future__ import annotations

import logging
import os
from typing import Any

from botocore.exceptions import ClientError

from aegis_scale.cloud.aws_catalog import default_us_east_1_pools
from aegis_scale.cloud.reconciler import AWSFleetReconciler, DynamoDBStateStore
from aegis_scale.config import SLAPolicy
from aegis_scale.engine import AegisScaleEngine
from aegis_scale.math_core.conformal import AdaptiveConformalPredictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_CLUSTER_ID = os.environ.get("AEGIS_CLUSTER_ID", "prod-web-tier")
_TABLE_NAME = os.environ.get("AEGIS_STATE_TABLE", "")
_ASG_NAME = os.environ.get("AEGIS_ASG_NAME", "aegis-managed-asg")
_DRY_RUN = os.environ.get("AEGIS_DRY_RUN", "true").lower() != "false"


def handle_control_event(
    event: dict[str, Any],
    state_store: DynamoDBStateStore | None = None,
    reconciler: AWSFleetReconciler | None = None,
) -> dict[str, Any]:
    """Process either an EventBridge Scheduler tick or an EC2 Spot Interruption/Rebalance event."""
    pools = default_us_east_1_pools()
    policy = SLAPolicy()
    engine = AegisScaleEngine(pools=pools, policy=policy)

    store = state_store
    if store is None and _TABLE_NAME:
        store = DynamoDBStateStore(table_name=_TABLE_NAME)

    if store is not None:
        saved = store.load_engine_state(_CLUSTER_ID)
        if saved is not None:
            if "predictor_state" in saved:
                engine.predictor = AdaptiveConformalPredictor.from_state_dict(saved["predictor_state"])
            if "allocations" in saved:
                engine.current_allocations = {
                    str(k): int(v) for k, v in saved["allocations"].items()
                }

    detail_type = str(event.get("detail-type", ""))
    detail = event.get("detail", {}) or {}
    compromised: list[str] = []

    if detail_type in (
        "EC2 Spot Instance Interruption Warning",
        "EC2 Instance Rebalance Recommendation",
    ):
        inst_type = str(detail.get("instance-type", ""))
        az = str(detail.get("availability-zone", ""))
        if inst_type and az:
            compromised = engine.flag_pool_preemption_warning(
                instance_type=inst_type,
                az=az,
                elevated_prob=0.99 if "Interruption" in detail_type else 0.85,
            )

    observed_rps = float(event.get("observed_rps", detail.get("observed_rps", 1200.0)))
    report = engine.run_cycle(observed_rps=observed_rps)

    rec = reconciler or AWSFleetReconciler(asg_name=_ASG_NAME, dry_run=_DRY_RUN)
    reconcile_status = rec.apply_fleet_decision(report)

    if store is not None:
        store.save_cycle_state_and_audit(
            cluster_id=_CLUSTER_ID,
            predictor_state=engine.predictor.state_dict(),
            report=report,
        )

    print(rec.build_emf_log_record(_CLUSTER_ID, report))

    return {
        "statusCode": 200,
        "cluster_id": _CLUSTER_ID,
        "trigger": detail_type or "ScheduledTick",
        "compromised_pools": compromised or report.compromised_pools,
        "required_vcpus": round(report.required_vcpus, 3),
        "provisioned_vcpus": report.portfolio.total_provisioned_vcpus,
        "on_demand_vcpus": report.portfolio.on_demand_vcpus,
        "spot_vcpus": report.portfolio.spot_vcpus,
        "hourly_cost_usd": round(report.portfolio.hourly_cost_usd, 5),
        "estimated_savings_pct": round(report.portfolio.estimated_savings_pct, 2),
        "cvar_shortfall_vcpu": round(report.portfolio.cvar_shortfall_vcpu, 5),
        "expected_p99_queue_wait_ms": round(report.expected_p99_queue_wait_ms, 3),
        "solve_time_ms": round(report.portfolio.solve_time_ms, 3),
        "allocations": report.portfolio.allocations,
        "reconcile": reconcile_status,
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint."""
    try:
        return handle_control_event(event)
    except ClientError as exc:
        logger.error("AWS SDK ClientError during control loop: %s", exc)
        raise

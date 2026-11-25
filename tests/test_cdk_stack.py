"""AWS CDK CloudFormation template synthesis and security/resilience assertions."""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import assertions

from infra.stack import AegisScaleStack


def test_cdk_stack_synthesizes_with_hardened_aws_compute_and_serverless_defaults() -> None:
    app = cdk.App()
    stack = AegisScaleStack(
        app,
        "TestAegisStack",
        env=cdk.Environment(account="111122223333", region="us-east-1"),
    )
    template = assertions.Template.from_stack(stack)

    # 1. Launch Template (never deprecated LaunchConfiguration) enforcing IMDSv2 + hop limit 2 + encrypted EBS
    template.resource_count_is("AWS::AutoScaling::LaunchConfiguration", 0)
    template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": assertions.Match.object_like(
                {
                    "MetadataOptions": {
                        "HttpTokens": "required",
                        "HttpPutResponseHopLimit": 2,
                    },
                    "BlockDeviceMappings": [
                        assertions.Match.object_like(
                            {
                                "Ebs": assertions.Match.object_like({"Encrypted": True}),
                            }
                        )
                    ],
                }
            )
        },
    )

    # 2. AutoScalingGroup uses ELB health check, CapacityRebalance, and price-capacity-optimized Spot
    template.has_resource_properties(
        "AWS::AutoScaling::AutoScalingGroup",
        {
            "HealthCheckType": "ELB",
            "CapacityRebalance": True,
            "MixedInstancesPolicy": assertions.Match.object_like(
                {
                    "InstancesDistribution": assertions.Match.object_like(
                        {
                            "SpotAllocationStrategy": "price-capacity-optimized",
                        }
                    )
                }
            ),
        },
    )

    # 3. DynamoDB FinOps State Ledger has PITR enabled
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
        },
    )

    # 4. EventBridge Spot Interruption & Rebalance Rule + EventBridge Scheduler with DLQ
    template.has_resource_properties(
        "AWS::Events::Rule",
        {
            "EventPattern": {
                "source": ["aws.ec2"],
                "detail-type": [
                    "EC2 Spot Instance Interruption Warning",
                    "EC2 Instance Rebalance Recommendation",
                ],
            }
        },
    )
    template.resource_count_is("AWS::Scheduler::Schedule", 1)

"""Production AWS CDK stack for AegisScale Control Plane and Mixed-Instance EC2 Fleet."""

from __future__ import annotations

from pathlib import Path
import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_autoscaling as autoscaling,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_scheduler as scheduler,
    aws_sqs as sqs,
)
from constructs import Construct


class AegisScaleStack(Stack):
    """Provisions hardened EC2 Mixed-Instance fleet, DynamoDB FinOps ledger, and Optimizer Control Plane."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Multi-AZ VPC for resilient Spot diversification
        vpc = ec2.Vpc(
            self,
            "AegisVpc",
            max_azs=3,
            nat_gateways=1,
        )

        # 2. Hardened Instance Role (SSM Session Manager only, zero SSH keys)
        instance_role = iam.Role(
            self,
            "FleetInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
            ],
        )

        fleet_sg = ec2.SecurityGroup(
            self,
            "FleetSecurityGroup",
            vpc=vpc,
            description="Security group for AegisScale managed compute fleet",
            allow_all_outbound=True,
        )

        # 3. Launch Template enforcing IMDSv2 (hop limit 2 for containers) and encrypted EBS
        launch_template = ec2.LaunchTemplate(
            self,
            "HardenedLaunchTemplate",
            instance_type=ec2.InstanceType("m7g.xlarge"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(
                cpu_type=ec2.AmazonLinuxCpuType.ARM_64
            ),
            role=instance_role,
            security_group=fleet_sg,
            require_imdsv2=True,
            http_put_response_hop_limit=2,
            detailed_monitoring=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=30,
                        encrypted=True,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
        )

        # Cross-family override template (x86_64/ARM compatible overrides for diversified Spot pools)
        asg = autoscaling.AutoScalingGroup(
            self,
            "ManagedComputeASG",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            min_capacity=2,
            max_capacity=200,
            capacity_rebalance=True,
            default_instance_warmup=Duration.seconds(120),
            health_checks=autoscaling.HealthChecks.with_additional_checks(
                grace_period=Duration.seconds(300),
                additional_types=[autoscaling.AdditionalHealthCheckType.ELB],
            ),
            mixed_instances_policy=autoscaling.MixedInstancesPolicy(
                launch_template=launch_template,
                instances_distribution=autoscaling.InstancesDistribution(
                    on_demand_base_capacity=2,
                    on_demand_percentage_above_base_capacity=25,
                    spot_allocation_strategy=autoscaling.SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED,
                ),
                launch_template_overrides=[
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("m7g.xlarge")),
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("c7g.xlarge")),
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("r7g.xlarge")),
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("m7g.2xlarge")),
                ],
            ),
        )

        # ALB + Instance Target Group wired to ASG so ELB health checks operate end-to-end
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "WorkloadALB",
            vpc=vpc,
            internet_facing=True,
        )
        listener = alb.add_listener("HttpListener", port=80, open=True)
        listener.add_targets(
            "FleetTargets",
            port=8080,
            targets=[asg],
            deregistration_delay=Duration.seconds(120),
            health_check=elbv2.HealthCheck(
                path="/health",
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
                interval=Duration.seconds(15),
            ),
        )

        # 4. Encrypted DynamoDB State & FinOps Audit Ledger
        state_table = dynamodb.Table(
            self,
            "FinOpsStateLedger",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        # 5. Encrypted Dead-Letter Queue for Control Plane resilience
        control_dlq = sqs.Queue(
            self,
            "OptimizerControlDLQ",
            encryption=sqs.QueueEncryption.KMS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
        )

        src_dir = str(Path(__file__).resolve().parents[1] / "src")

        # 6. Optimizer Event Function (On-Demand Lambda)
        optimizer_fn = _lambda.Function(
            self,
            "AegisOptimizerFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="aegis_scale.cloud.lambda_handler.handler",
            code=_lambda.Code.from_asset(src_dir),
            memory_size=1024,
            timeout=Duration.seconds(30),
            dead_letter_queue=control_dlq,
            log_retention=logs.RetentionDays.ONE_MONTH,
            environment={
                "AEGIS_CLUSTER_ID": "prod-web-tier",
                "AEGIS_STATE_TABLE": state_table.table_name,
                "AEGIS_ASG_NAME": asg.auto_scaling_group_name,
                "AEGIS_DRY_RUN": "false",
            },
        )

        # Least-privilege IAM permissions
        state_table.grant_read_write_data(optimizer_fn)
        optimizer_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ec2:DescribeSpotPriceHistory"],
                resources=["*"],
            )
        )
        optimizer_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["autoscaling:UpdateAutoScalingGroup"],
                resources=[
                    cdk.Arn.format(
                        cdk.ArnComponents(
                            service="autoscaling",
                            resource="autoScalingGroup",
                            resource_name=f"*autoScalingGroupName/{asg.auto_scaling_group_name}",
                            arn_format=cdk.ArnFormat.COLON_RESOURCE_NAME,
                        ),
                        self,
                    )
                ],
            )
        )

        # 7. Reactive EventBridge Rule: EC2 2-min Spot Interruption & Rebalance Recommendation
        spot_warning_rule = events.Rule(
            self,
            "SpotInterruptionAndRebalanceRule",
            description="Sub-second evacuation trigger on EC2 Spot interruption or rebalance signals",
            event_pattern=events.EventPattern(
                source=["aws.ec2"],
                detail_type=[
                    "EC2 Spot Instance Interruption Warning",
                    "EC2 Instance Rebalance Recommendation",
                ],
            ),
        )
        spot_warning_rule.add_target(
            targets.LambdaFunction(
                optimizer_fn,
                dead_letter_queue=control_dlq,
                retry_attempts=2,
            )
        )

        # 8. EventBridge Scheduler (1-minute optimization cadence with retry + DLQ)
        scheduler_role = iam.Role(
            self,
            "OptimizerSchedulerRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
        )
        optimizer_fn.grant_invoke(scheduler_role)
        control_dlq.grant_send_messages(scheduler_role)

        scheduler.CfnSchedule(
            self,
            "MinuteOptimizationSchedule",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF"),
            schedule_expression="rate(1 minute)",
            target=scheduler.CfnSchedule.TargetProperty(
                arn=optimizer_fn.function_arn,
                role_arn=scheduler_role.role_arn,
                dead_letter_config=scheduler.CfnSchedule.DeadLetterConfigProperty(
                    arn=control_dlq.queue_arn
                ),
                retry_policy=scheduler.CfnSchedule.RetryPolicyProperty(
                    maximum_event_age_in_seconds=60,
                    maximum_retry_attempts=2,
                ),
            ),
        )

        cdk.CfnOutput(self, "StateLedgerTableName", value=state_table.table_name)
        cdk.CfnOutput(self, "ManagedASGName", value=asg.auto_scaling_group_name)
        cdk.CfnOutput(self, "WorkloadALBDns", value=alb.load_balancer_dns_name)

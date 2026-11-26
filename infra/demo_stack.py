"""Serverless zero-idle-cost AWS CDK stack for hosting the live AegisScale demo on a custom subdomain.

Deploys the FastAPI + HiGHS MILP simulator as an AWS Lambda Container Image (Event Function
with AWS Lambda Web Adapter) fronted by Amazon API Gateway HTTP API with strict rate/burst
throttling (bill protection) and optional Regional Custom Domain (e.g., aegis.zenithcodigo.com)
compatible with external domain registrars (Namecheap, GoDaddy, Hostinger, OVH).
"""

from __future__ import annotations

from pathlib import Path
import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as apigwv2_integrations,
    aws_certificatemanager as acm,
    aws_lambda as _lambda,
    aws_logs as logs,
)
from constructs import Construct


class AegisLiveDemoStack(Stack):
    """Scale-to-zero serverless stack for https://aegis.zenithcodigo.com."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        domain_name: str | None = None,
        certificate_arn: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        repo_root = str(Path(__file__).resolve().parents[1])

        # 1. Serverless Container Event Function (scales to $0 at idle; 1536 MB for fast HiGHS SIMD)
        demo_fn = _lambda.DockerImageFunction(
            self,
            "AegisDemoWebFunction",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=repo_root,
                file="Dockerfile.lambda",
                exclude=[".git", ".venv", "cdk.out", "output", "paper", "tests"],
            ),
            architecture=_lambda.Architecture.X86_64,
            memory_size=1536,
            timeout=Duration.seconds(29),
            log_retention=logs.RetentionDays.TWO_WEEKS,
            environment={
                "AEGIS_CLUSTER_ID": "public-live-demo",
                "AEGIS_DRY_RUN": "true",
                "AEGIS_PUBLIC_DEMO": "true",
            },
        )

        integration = apigwv2_integrations.HttpLambdaIntegration(
            "FastApiLambdaIntegration",
            handler=demo_fn,
        )

        # 2. Optional Custom Domain Mapping (for external registrar CNAME -> API Gateway Regional target)
        custom_domain_mapping = None
        apigw_domain = None
        if domain_name and certificate_arn:
            cert = acm.Certificate.from_certificate_arn(
                self, "ExternalRegistrarCert", certificate_arn
            )
            apigw_domain = apigwv2.DomainName(
                self,
                "DemoCustomDomain",
                domain_name=domain_name,
                certificate=cert,
            )
            custom_domain_mapping = apigwv2.DomainMappingOptions(domain_name=apigw_domain)

        # 3. API Gateway HTTP API (~70% lower per-request cost than REST API)
        http_api = apigwv2.HttpApi(
            self,
            "AegisDemoHttpApi",
            description="Public scale-to-zero HTTP API for AegisScale simulator",
            default_integration=integration,
            default_domain_mapping=custom_domain_mapping,
        )

        # Enforce stage-level rate & burst throttling to protect against traffic abuse
        default_stage = http_api.default_stage
        if default_stage is not None:
            cfn_stage = default_stage.node.default_child
            if isinstance(cfn_stage, apigwv2.CfnStage):
                cfn_stage.default_route_settings = apigwv2.CfnStage.RouteSettingsProperty(
                    throttling_rate_limit=3,
                    throttling_burst_limit=8,
                )

        cdk.CfnOutput(
            self,
            "DefaultHttpApiUrl",
            value=http_api.api_endpoint,
            description="Direct HTTPS API Gateway endpoint",
        )

        if apigw_domain is not None:
            cdk.CfnOutput(
                self,
                "RegistrarCnameTarget",
                value=apigw_domain.regional_domain_name,
                description=f"Point CNAME '{domain_name}' in your registrar DNS to this target",
            )

"""AWS CDK application entrypoint for AegisScale (Production Fleet + Scale-to-Zero Live Demo)."""

import os
import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from infra.demo_stack import AegisLiveDemoStack
from infra.stack import AegisScaleStack


def build_cdk_app() -> cdk.App:
    app = cdk.App()
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT", "111122223333"),
        region=os.environ.get("CDK_DEFAULT_REGION", "eu-west-1"),
    )
    AegisScaleStack(app, "AegisScaleProdStack", env=env)

    domain_name = app.node.try_get_context("domain_name")
    cert_arn = app.node.try_get_context("cert_arn")
    AegisLiveDemoStack(
        app,
        "AegisLiveDemoStack",
        domain_name=domain_name,
        certificate_arn=cert_arn,
        env=env,
    )
    return app


if __name__ == "__main__":
    cdk_app = build_cdk_app()
    cdk.Aspects.of(cdk_app).add(AwsSolutionsChecks(verbose=False))
    cdk_app.synth()

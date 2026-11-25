"""AWS CDK application entrypoint for AegisScale."""

import os
import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from infra.stack import AegisScaleStack


def build_cdk_app() -> cdk.App:
    app = cdk.App()
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT", "111122223333"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    )
    AegisScaleStack(app, "AegisScaleProdStack", env=env)
    return app


if __name__ == "__main__":
    cdk_app = build_cdk_app()
    cdk.Aspects.of(cdk_app).add(AwsSolutionsChecks(verbose=False))
    cdk_app.synth()

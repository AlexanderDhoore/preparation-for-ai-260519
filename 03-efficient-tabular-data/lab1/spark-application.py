from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/03-efficient-tabular-data")

from s3_tabular_common import (  # noqa: E402
    load_environment,
    participant_bucket,
    require_env,
)

OUTPUT_DIR = Path("/root/preparation-for-ai/03-efficient-tabular-data/lab1")

# TODO:
# Choose where Spark should write its metric table in your participant bucket.
SPARK_OUTPUT_PREFIX = "TODO_FILL_SPARK_OUTPUT_PREFIX"

SILVER_PREFIX = "silver/chapter03/covertype"
SPARK_SCRIPT = Path(
    "/root/preparation-for-ai/03-efficient-tabular-data/spark_chapter03_metrics.py"
)
SPARK_WORKLOAD_SERVICE_ACCOUNT = "spark"
SPARK_IMAGE = "apache/spark:4.1.1-python3"
SPARK_VERSION = "4.1.1"


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise NotImplementedError(message)


def normalize_endpoint(endpoint: str) -> str:
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    if not verify_ssl and endpoint.startswith("https://"):
        return "http://" + endpoint.removeprefix("https://")
    return endpoint


def kubernetes_name(value: str, prefix: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    if not cleaned:
        cleaned = "participant"
    if not prefix:
        return cleaned[:63].rstrip("-")
    return f"{prefix}-{cleaned}"[:63].rstrip("-")


def indent_block(text: str, spaces: int) -> str:
    return textwrap.indent(text.rstrip() + "\n", " " * spaces)


def build_manifest() -> tuple[str, dict[str, str]]:
    load_environment()
    require_completed(SPARK_OUTPUT_PREFIX, "TODO: fill in SPARK_OUTPUT_PREFIX.")

    bucket = participant_bucket()
    namespace = kubernetes_name(bucket, "")
    secret_name = namespace
    app_name = "chapter03-spark"
    config_map_name = "chapter03-spark-code"
    endpoint = normalize_endpoint(require_env("S3_ENDPOINT_URL"))
    source_uri = f"s3a://{bucket}/{SILVER_PREFIX}/"
    output_uri = f"s3a://{bucket}/{SPARK_OUTPUT_PREFIX.rstrip('/')}/"
    script = SPARK_SCRIPT.read_text(encoding="utf-8")

    manifest = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config_map_name}
  namespace: {namespace}
data:
  spark_chapter03_metrics.py: |
{indent_block(script, 4)}---
apiVersion: spark.apache.org/v1
kind: SparkApplication
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  pyFiles: "local:///opt/spark/work-dir/app/spark_chapter03_metrics.py"
  driverArgs:
    - "{source_uri}"
    - "{output_uri}"
  sparkConf:
    spark.logConf: "true"
    spark.driver.memory: "1g"
    spark.executor.memory: "1g"
    spark.executor.instances: "1"
    spark.executor.cores: "1"
    spark.jars.packages: "org.apache.hadoop:hadoop-aws:3.4.2"
    spark.jars.ivy: "/tmp/.ivy2.5.2"
    spark.kubernetes.authenticate.driver.serviceAccountName: "{SPARK_WORKLOAD_SERVICE_ACCOUNT}"
    spark.kubernetes.container.image: "{SPARK_IMAGE}"
    spark.kubernetes.driver.secretKeyRef.AWS_ACCESS_KEY_ID: "{secret_name}:AWS_ACCESS_KEY_ID"
    spark.kubernetes.driver.secretKeyRef.AWS_SECRET_ACCESS_KEY: "{secret_name}:AWS_SECRET_ACCESS_KEY"
    spark.kubernetes.driverEnv.S3_ENDPOINT_URL: "{endpoint}"
    spark.kubernetes.driver.podTemplateContainerName: "spark-kubernetes-driver"
    spark.kubernetes.executor.secretKeyRef.AWS_ACCESS_KEY_ID: "{secret_name}:AWS_ACCESS_KEY_ID"
    spark.kubernetes.executor.secretKeyRef.AWS_SECRET_ACCESS_KEY: "{secret_name}:AWS_SECRET_ACCESS_KEY"
    spark.executorEnv.S3_ENDPOINT_URL: "{endpoint}"
    spark.kubernetes.executor.podTemplateContainerName: "spark-kubernetes-executor"
  driverSpec:
    podTemplateSpec:
      spec:
        containers:
          - name: spark-kubernetes-driver
            volumeMounts:
              - name: chapter03-spark-code
                mountPath: /opt/spark/work-dir/app
                readOnly: true
        volumes:
          - name: chapter03-spark-code
            configMap:
              name: {config_map_name}
  executorSpec:
    podTemplateSpec:
      spec:
        containers:
          - name: spark-kubernetes-executor
            volumeMounts:
              - name: chapter03-spark-code
                mountPath: /opt/spark/work-dir/app
                readOnly: true
        volumes:
          - name: chapter03-spark-code
            configMap:
              name: {config_map_name}
  applicationTolerations:
    resourceRetainPolicy: Always
  runtimeVersions:
    sparkVersion: "{SPARK_VERSION}"
"""

    metadata = {
        "app_name": app_name,
        "config_map_name": config_map_name,
        "namespace": namespace,
        "source_uri": source_uri,
        "output_uri": output_uri,
        "spark_image": SPARK_IMAGE,
        "spark_version": SPARK_VERSION,
        "credential_secret": secret_name,
        "spark_service_account": SPARK_WORKLOAD_SERVICE_ACCOUNT,
    }
    return manifest, metadata


def main() -> None:
    manifest, metadata = build_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "spark1_application.yaml"
    manifest_path.write_text(manifest, encoding="utf-8")

    print(f"Wrote SparkApplication manifest: {manifest_path}")
    print(f"Application name: {metadata['app_name']}")
    print(f"Namespace: {metadata['namespace']}")
    print(f"Source Parquet: {metadata['source_uri']}")
    print(f"Output Parquet: {metadata['output_uri']}")
    print("")
    print("Apply it with:")
    print(f"  KUBECONFIG=participant.kubeconfig kubectl apply -f {manifest_path}")
    print("")
    print("Watch it with:")
    print(
        "  KUBECONFIG=participant.kubeconfig "
        f"kubectl -n {metadata['namespace']} get sparkapp {metadata['app_name']} -w"
    )


if __name__ == "__main__":
    main()

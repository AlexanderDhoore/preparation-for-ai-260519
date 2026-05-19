from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from urllib.parse import urlparse

import boto3
import polars as pl
from dotenv import load_dotenv
from pyiceberg.catalog import load_catalog

REPO_ROOT = Path("/root/preparation-for-ai")
CHAPTER_ROOT = Path("/root/preparation-for-ai/04-open-table-formats")

SPARK_WORKLOAD_SERVICE_ACCOUNT = "spark"
SPARK_IMAGE = "apache/spark:4.1.1-python3"
SPARK_VERSION = "4.1.1"
ICEBERG_VERSION = "1.10.1"
ICEBERG_RUNTIME_PACKAGE = (
    f"org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:{ICEBERG_VERSION}"
)
ICEBERG_AWS_PACKAGE = f"org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}"
HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.4.2"

SHARED_TAXI_PREFIX = "bronze/chapter04/nyc-taxi"
ICEBERG_WAREHOUSE_PREFIX = "chapter04/iceberg-warehouse"
DEFAULT_CATALOG_ALIAS = "course"
DEFAULT_POLARIS_CATALOG = "preparation_for_ai"
DEFAULT_POLARIS_SCOPE = "PRINCIPAL_ROLE:ALL"
TAXI_LAB_SAMPLE_ROWS = 1_000_000

TAXI_MONTH_FILES = {
    "2024-01": "yellow_taxi_2024-01_sample.parquet",
    "2024-02": "yellow_taxi_2024-02_sample.parquet",
}

TAXI_SOURCE_MONTH_FILES = {
    "2024-01": "yellow_tripdata_2024-01.parquet",
    "2024-02": "yellow_tripdata_2024-02.parquet",
}


def load_environment() -> None:
    load_dotenv(REPO_ROOT / "participant.env")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise NotImplementedError(message)


def participant_id() -> str:
    load_environment()
    return require_env("PARTICIPANT_ID")


def participant_bucket() -> str:
    load_environment()
    return os.getenv("PARTICIPANT_BUCKET") or require_env("S3_BUCKET")


def shared_bucket() -> str:
    load_environment()
    return os.getenv("SHARED_BUCKET") or require_env("S3_SHARED_BUCKET")


def normalize_endpoint(endpoint: str) -> str:
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    if not verify_ssl and endpoint.startswith("https://"):
        return "http://" + endpoint.removeprefix("https://")
    return endpoint


def s3_endpoint_for_duckdb() -> tuple[str, bool]:
    endpoint = normalize_endpoint(require_env("S3_ENDPOINT_URL"))
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit("S3_ENDPOINT_URL should look like http://host or https://host")
    return parsed.netloc, parsed.scheme == "https"


def s3_client():
    load_environment()
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    return boto3.client(
        "s3",
        endpoint_url=require_env("S3_ENDPOINT_URL"),
        aws_access_key_id=require_env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("S3_SECRET_ACCESS_KEY"),
        verify=verify_ssl,
    )


def delete_prefix(client, bucket: str, prefix: str) -> int:
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if not objects:
            continue
        client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        deleted += len(objects)
    return deleted


def polars_storage_options() -> dict[str, str]:
    load_environment()
    endpoint = normalize_endpoint(require_env("S3_ENDPOINT_URL"))
    options = {
        "aws_access_key_id": require_env("S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": require_env("S3_SECRET_ACCESS_KEY"),
        "aws_endpoint_url": endpoint,
        "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if endpoint.startswith("http://"):
        options["aws_allow_http"] = "true"
    return options


def kubernetes_name(value: str, prefix: str = "") -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    if not cleaned:
        cleaned = "participant"
    if prefix:
        cleaned = f"{prefix}-{cleaned}"
    return cleaned[:63].rstrip("-")


def sql_identifier(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "participant"
    if cleaned[0].isdigit():
        cleaned = f"participant_{cleaned}"
    return cleaned


def sql_identifier_suffix(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "participant"
    return cleaned


def require_sql_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise SystemExit(
            f"{label} should be a simple lowercase SQL identifier, "
            "for example preparation_for_ai_01 or yellow_taxi_trips."
        )


def participant_kubernetes_namespace() -> str:
    return kubernetes_name(participant_bucket())


def iceberg_namespace() -> str:
    load_environment()
    explicit = os.getenv("ICEBERG_NAMESPACE")
    if explicit:
        require_sql_identifier(explicit, "ICEBERG_NAMESPACE")
        return explicit
    return f"preparation_for_ai_{sql_identifier_suffix(participant_id())}"


def iceberg_catalog_alias() -> str:
    load_environment()
    alias = os.getenv("ICEBERG_CATALOG_ALIAS") or os.getenv(
        "ICEBERG_CATALOG_NAME", DEFAULT_CATALOG_ALIAS
    )
    require_sql_identifier(alias, "ICEBERG_CATALOG_ALIAS")
    return alias


def polaris_catalog_name() -> str:
    load_environment()
    name = os.getenv("POLARIS_CATALOG_NAME", DEFAULT_POLARIS_CATALOG)
    require_sql_identifier(name, "POLARIS_CATALOG_NAME")
    return name


def polaris_scope() -> str:
    load_environment()
    return os.getenv("POLARIS_SCOPE", DEFAULT_POLARIS_SCOPE)


def polaris_rest_uri(*, internal: bool) -> str:
    load_environment()
    if internal:
        return os.getenv("ICEBERG_REST_INTERNAL_URI") or require_env("ICEBERG_REST_URI")
    return require_env("ICEBERG_REST_URI")


def polaris_oauth_token_uri(*, internal: bool) -> str:
    load_environment()
    explicit_name = (
        "POLARIS_OAUTH_TOKEN_INTERNAL_URI" if internal else "POLARIS_OAUTH_TOKEN_URI"
    )
    if explicit := os.getenv(explicit_name):
        return explicit
    return polaris_rest_uri(internal=internal).rstrip("/") + "/v1/oauth/tokens"


def pyiceberg_catalog():
    endpoint = normalize_endpoint(require_env("S3_ENDPOINT_URL"))
    return load_catalog(
        iceberg_catalog_alias(),
        type="rest",
        uri=polaris_rest_uri(internal=False),
        warehouse=polaris_catalog_name(),
        credential=f"{require_env('POLARIS_CLIENT_ID')}:{require_env('POLARIS_CLIENT_SECRET')}",
        scope=polaris_scope(),
        **{
            "oauth2-server-uri": polaris_oauth_token_uri(internal=False),
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.endpoint": endpoint,
            "s3.access-key-id": require_env("S3_ACCESS_KEY_ID"),
            "s3.secret-access-key": require_env("S3_SECRET_ACCESS_KEY"),
            "s3.region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "s3.force-virtual-addressing": "false",
            "s3.resolve-region": "false",
            "header.X-Iceberg-Access-Delegation": "",
        },
    )


def s3a_uri(bucket: str, key_or_prefix: str) -> str:
    return f"s3a://{bucket}/{key_or_prefix.strip('/')}"


def s3_uri(bucket: str, key_or_prefix: str) -> str:
    return f"s3://{bucket}/{key_or_prefix.strip('/')}"


def shared_taxi_key(month: str) -> str:
    try:
        filename = TAXI_MONTH_FILES[month]
    except KeyError as exc:
        raise SystemExit(f"Unknown taxi month: {month}") from exc
    return f"{SHARED_TAXI_PREFIX}/{filename}"


def shared_taxi_uri(month: str) -> str:
    return s3_uri(shared_bucket(), shared_taxi_key(month))


def shared_taxi_s3a_uri(month: str) -> str:
    return s3a_uri(shared_bucket(), shared_taxi_key(month))


def iceberg_table_identifier(table_name: str) -> str:
    return f"{iceberg_namespace()}.{table_name}"


def full_table_name(table_name: str) -> str:
    return f"{iceberg_catalog_alias()}.{iceberg_table_identifier(table_name)}"


def iceberg_table_location(table_name: str) -> str:
    bucket = participant_bucket()
    prefix = f"{ICEBERG_WAREHOUSE_PREFIX}/{iceberg_namespace()}/{table_name}"
    return s3_uri(bucket, prefix)


def iceberg_namespace_location() -> str:
    bucket = participant_bucket()
    prefix = f"{ICEBERG_WAREHOUSE_PREFIX}/{iceberg_namespace()}"
    return s3_uri(bucket, prefix)


def iceberg_table_location_prefix(table_name: str) -> str:
    bucket = participant_bucket()
    prefix = f"{ICEBERG_WAREHOUSE_PREFIX}/{iceberg_namespace()}/{table_name}"
    return s3_uri(bucket, prefix) + "/"


def clean_taxi_lazy_frame(month: str, *, limit_rows: int | None = None):
    source = shared_taxi_uri(month)
    frame = pl.scan_parquet(source, storage_options=polars_storage_options())
    if limit_rows is not None:
        frame = frame.limit(limit_rows)
    return frame


def source_taxi_arrow_table(month: str = "2024-01"):
    return clean_taxi_lazy_frame(month).collect().to_arrow()


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def indent_block(text: str, spaces: int) -> str:
    return textwrap.indent(text.rstrip() + "\n", " " * spaces)


def build_spark_application(
    *,
    app_name: str,
    config_map_name: str,
    script_name: str,
    driver_args: list[str],
    report_name: str,
) -> tuple[str, dict[str, str]]:
    load_environment()
    bucket = participant_bucket()
    namespace = participant_kubernetes_namespace()
    secret_name = namespace
    endpoint = normalize_endpoint(require_env("S3_ENDPOINT_URL"))
    catalog_alias = iceberg_catalog_alias()
    polaris_catalog = polaris_catalog_name()
    catalog_url = polaris_rest_uri(internal=True)
    oauth_token_url = polaris_oauth_token_uri(internal=True)
    scope = polaris_scope()
    script_path = CHAPTER_ROOT / script_name
    script = script_path.read_text(encoding="utf-8")
    packages = ",".join(
        [ICEBERG_RUNTIME_PACKAGE, ICEBERG_AWS_PACKAGE, HADOOP_AWS_PACKAGE]
    )

    args_yaml = "\n".join(f'    - "{arg}"' for arg in driver_args)
    manifest = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config_map_name}
  namespace: {namespace}
data:
  {script_name}: |
{indent_block(script, 4)}---
apiVersion: spark.apache.org/v1
kind: SparkApplication
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  pyFiles: "local:///opt/spark/work-dir/app/{script_name}"
  driverArgs:
{args_yaml}
  sparkConf:
    spark.logConf: "true"
    spark.redaction.regex: "(?i)secret|password|token|credential|access[.]?key"
    spark.driver.memory: "2g"
    spark.executor.memory: "2g"
    spark.executor.instances: "1"
    spark.executor.cores: "1"
    spark.jars.packages: "{packages}"
    spark.jars.ivy: "/tmp/.ivy2.5.2"
    spark.sql.extensions: "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
    spark.sql.catalog.{catalog_alias}: "org.apache.iceberg.spark.SparkCatalog"
    spark.sql.catalog.{catalog_alias}.type: "rest"
    spark.sql.catalog.{catalog_alias}.uri: "{catalog_url}"
    spark.sql.catalog.{catalog_alias}.warehouse: "{polaris_catalog}"
    spark.sql.catalog.{catalog_alias}.oauth2-server-uri: "{oauth_token_url}"
    spark.sql.catalog.{catalog_alias}.header.X-Iceberg-Access-Delegation: ""
    spark.sql.catalog.{catalog_alias}.io-impl: "org.apache.iceberg.aws.s3.S3FileIO"
    spark.sql.catalog.{catalog_alias}.token-refresh-enabled: "true"
    spark.sql.catalog.{catalog_alias}.client.region: "irrelevant"
    spark.sql.catalog.{catalog_alias}.s3.endpoint: "{endpoint}"
    spark.sql.catalog.{catalog_alias}.s3.path-style-access: "true"
    spark.hadoop.fs.s3a.impl: "org.apache.hadoop.fs.s3a.S3AFileSystem"
    spark.hadoop.fs.s3a.endpoint: "{endpoint}"
    spark.hadoop.fs.s3a.path.style.access: "true"
    spark.hadoop.fs.s3a.connection.ssl.enabled: "{'true' if endpoint.startswith('https://') else 'false'}"
    spark.kubernetes.authenticate.driver.serviceAccountName: "{SPARK_WORKLOAD_SERVICE_ACCOUNT}"
    spark.kubernetes.container.image: "{SPARK_IMAGE}"
    spark.kubernetes.driver.secretKeyRef.AWS_ACCESS_KEY_ID: "{secret_name}:AWS_ACCESS_KEY_ID"
    spark.kubernetes.driver.secretKeyRef.AWS_SECRET_ACCESS_KEY: "{secret_name}:AWS_SECRET_ACCESS_KEY"
    spark.kubernetes.driver.secretKeyRef.POLARIS_CLIENT_ID: "{secret_name}:POLARIS_CLIENT_ID"
    spark.kubernetes.driver.secretKeyRef.POLARIS_CLIENT_SECRET: "{secret_name}:POLARIS_CLIENT_SECRET"
    spark.kubernetes.driverEnv.S3_ENDPOINT_URL: "{endpoint}"
    spark.kubernetes.driverEnv.POLARIS_SCOPE: "{scope}"
    spark.kubernetes.driver.podTemplateContainerName: "spark-kubernetes-driver"
    spark.kubernetes.executor.secretKeyRef.AWS_ACCESS_KEY_ID: "{secret_name}:AWS_ACCESS_KEY_ID"
    spark.kubernetes.executor.secretKeyRef.AWS_SECRET_ACCESS_KEY: "{secret_name}:AWS_SECRET_ACCESS_KEY"
    spark.kubernetes.executor.secretKeyRef.POLARIS_CLIENT_ID: "{secret_name}:POLARIS_CLIENT_ID"
    spark.kubernetes.executor.secretKeyRef.POLARIS_CLIENT_SECRET: "{secret_name}:POLARIS_CLIENT_SECRET"
    spark.executorEnv.S3_ENDPOINT_URL: "{endpoint}"
    spark.executorEnv.POLARIS_SCOPE: "{scope}"
    spark.kubernetes.executor.podTemplateContainerName: "spark-kubernetes-executor"
  driverSpec:
    podTemplateSpec:
      spec:
        containers:
          - name: spark-kubernetes-driver
            volumeMounts:
              - name: {config_map_name}
                mountPath: /opt/spark/work-dir/app
                readOnly: true
        volumes:
          - name: {config_map_name}
            configMap:
              name: {config_map_name}
  executorSpec:
    podTemplateSpec:
      spec:
        containers:
          - name: spark-kubernetes-executor
            volumeMounts:
              - name: {config_map_name}
                mountPath: /opt/spark/work-dir/app
                readOnly: true
        volumes:
          - name: {config_map_name}
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
        "bucket": bucket,
        "catalog_alias": catalog_alias,
        "polaris_catalog": polaris_catalog,
        "catalog_url": catalog_url,
        "script_name": script_name,
        "spark_image": SPARK_IMAGE,
        "spark_version": SPARK_VERSION,
        "iceberg_version": ICEBERG_VERSION,
        "credential_secret": secret_name,
        "spark_service_account": SPARK_WORKLOAD_SERVICE_ACCOUNT,
    }

    output_dir = CHAPTER_ROOT / "lab1"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{report_name}.yaml"
    report_path = output_dir / f"{report_name}_report.md"
    manifest_path.write_text(manifest, encoding="utf-8")
    report_path.write_text(
        spark_manifest_report(metadata, manifest_path), encoding="utf-8"
    )

    return str(manifest_path), metadata


def spark_manifest_report(metadata: dict[str, str], manifest_path: Path) -> str:
    return "\n".join(
        [
            "# Spark Iceberg Manifest Report",
            "",
            f"Manifest: `{manifest_path}`",
            f"SparkApplication: `{metadata['app_name']}`",
            f"Kubernetes namespace: `{metadata['namespace']}`",
            f"Spark image: `{metadata['spark_image']}`",
            f"Iceberg version: `{metadata['iceberg_version']}`",
            f"Polaris catalog: `{metadata['polaris_catalog']}`",
            f"Credential secret: `{metadata['credential_secret']}`",
            "",
        ]
    )


def print_manifest_instructions(manifest_path: str, metadata: dict[str, str]) -> None:
    print(f"Wrote SparkApplication manifest: {manifest_path}")
    print("")
    print("Apply it with:")
    print(f"  KUBECONFIG=participant.kubeconfig kubectl apply -f {manifest_path}")
    print("")
    print("Watch it with:")
    print(
        "  KUBECONFIG=participant.kubeconfig "
        f"kubectl -n {metadata['namespace']} get sparkapp {metadata['app_name']} -w"
    )

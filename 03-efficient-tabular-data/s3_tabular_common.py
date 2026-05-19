from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv

REPO_ROOT = Path("/root/preparation-for-ai")


def load_environment() -> None:
    load_dotenv(REPO_ROOT / "participant.env")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def participant_bucket() -> str:
    load_environment()
    return os.getenv("PARTICIPANT_BUCKET") or require_env("S3_BUCKET")


def shared_bucket() -> str:
    load_environment()
    return os.getenv("SHARED_BUCKET") or require_env("S3_SHARED_BUCKET")


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


def polars_storage_options() -> dict[str, str]:
    load_environment()
    endpoint = require_env("S3_ENDPOINT_URL")
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    if not verify_ssl and endpoint.startswith("https://"):
        endpoint = "http://" + endpoint.removeprefix("https://")

    options = {
        "aws_access_key_id": require_env("S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": require_env("S3_SECRET_ACCESS_KEY"),
        "aws_endpoint_url": endpoint,
        "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if endpoint.startswith("http://"):
        options["aws_allow_http"] = "true"
    return options


def configure_duckdb_s3(connection) -> None:
    load_environment()
    endpoint = require_env("S3_ENDPOINT_URL")
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    if not verify_ssl and endpoint.startswith("https://"):
        endpoint = "http://" + endpoint.removeprefix("https://")
    use_ssl = endpoint.startswith("https://")
    endpoint = endpoint.removeprefix("https://").removeprefix("http://")
    key_id = require_env("S3_ACCESS_KEY_ID").replace("'", "''")
    secret = require_env("S3_SECRET_ACCESS_KEY").replace("'", "''")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1").replace("'", "''")
    connection.sql("INSTALL httpfs; LOAD httpfs;")
    connection.sql(
        f"""
        CREATE OR REPLACE SECRET s3_course (
            TYPE S3,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION '{region}',
            ENDPOINT '{endpoint}',
            URL_STYLE 'path',
            USE_SSL {'true' if use_ssl else 'false'}
        );
        """
    )


def s3_uri(bucket: str, key_or_prefix: str) -> str:
    return f"s3://{bucket}/{key_or_prefix.lstrip('/')}"


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


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return sorted(keys)


def upload_json(client, bucket: str, key: str, payload: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )

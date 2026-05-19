from __future__ import annotations

import os
import shutil
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import urllib3
from dotenv import load_dotenv

REPO_ROOT = Path("/root/preparation-for-ai")
LAB_DIR = REPO_ROOT / "06-orchestration" / "lab1"
CACHE_DIR = LAB_DIR / "cache"

# TODO:
# Fill in the run prefix printed by Airflow.
# Example:
# RUN_PREFIX = "outputs/chapter06/taxi-python/run_id=manual-2026-05-18t12-00-00/"
RUN_PREFIX = "TODO_FILL_AIRFLOW_RUN_PREFIX"


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise SystemExit(message)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or "replace-me" in value or "example.invalid" in value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def s3_client():
    load_dotenv(REPO_ROOT / "participant.env")
    verify_ssl = os.getenv("S3_VERIFY_SSL", "true").strip().lower()
    verify = verify_ssl not in {"0", "false", "no"}
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return boto3.client(
        "s3",
        endpoint_url=require_env("S3_ENDPOINT_URL"),
        aws_access_key_id=require_env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("S3_SECRET_ACCESS_KEY"),
        verify=verify,
    )


def participant_bucket() -> str:
    load_dotenv(REPO_ROOT / "participant.env")
    return require_env("PARTICIPANT_BUCKET")


def list_objects(client, bucket: str, prefix: str) -> list[dict[str, object]]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, object]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return sorted(objects, key=lambda item: item["Key"])


def download_prefix(client, bucket: str, prefix: str, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in list_objects(client, bucket, prefix):
        key = item["Key"]
        if key.endswith("/") or key.endswith("_SUCCESS"):
            continue
        local_path = target_dir / Path(key).name
        client.download_file(bucket, key, str(local_path))


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing expected output file: {path}")
    return pd.read_parquet(path)


def save_hourly_plot(hourly: pd.DataFrame) -> None:
    hourly = hourly.sort_values("pickup_hour")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plots = [
        ("trips", "Trips", "#356f8c"),
        ("average_distance", "Average distance", "#6a994e"),
        ("average_total", "Average total amount", "#bc6c25"),
        ("average_tip_rate", "Average tip rate", "#c43b3b"),
    ]
    for ax, (column, title, color) in zip(axes.ravel(), plots):
        ax.plot(hourly["pickup_hour"], hourly[column], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("Pickup hour")
        ax.set_xticks(range(0, 24, 3))
    fig.suptitle("Taxi metrics by pickup hour")
    fig.tight_layout()
    fig.savefig(LAB_DIR / "taxi2_hourly_metrics.png", dpi=150)
    plt.close(fig)


def save_payment_plot(payment: pd.DataFrame) -> None:
    payment = payment.sort_values("trips", ascending=True)
    labels = payment["payment_type_label"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plots = [
        ("trips", "Trips", "#356f8c"),
        ("average_distance", "Average distance", "#6a994e"),
        ("average_total", "Average total amount", "#bc6c25"),
        ("average_tip_rate", "Average tip rate", "#c43b3b"),
    ]
    for ax, (column, title, color) in zip(axes.ravel(), plots):
        ax.barh(labels, payment[column], color=color)
        ax.set_title(title)
    fig.suptitle("Taxi metrics by payment type")
    fig.tight_layout()
    fig.savefig(LAB_DIR / "taxi3_payment_metrics.png", dpi=150)
    plt.close(fig)


def save_distance_plot(distance: pd.DataFrame) -> None:
    order = ["0-1 miles", "1-3 miles", "3-8 miles", "8-20 miles", "20+ miles"]
    distance = distance.assign(
        distance_bucket=pd.Categorical(
            distance["distance_bucket"],
            categories=order,
            ordered=True,
        )
    ).sort_values("distance_bucket")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plots = [
        ("trips", "Trips", "#356f8c"),
        ("average_total", "Average total amount", "#bc6c25"),
        ("average_tip_rate", "Average tip rate", "#c43b3b"),
    ]
    for ax, (column, title, color) in zip(axes, plots):
        ax.bar(distance["distance_bucket"].astype(str), distance[column], color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Taxi metrics by distance bucket")
    fig.tight_layout()
    fig.savefig(LAB_DIR / "taxi4_distance_metrics.png", dpi=150)
    plt.close(fig)


def write_summary(objects: list[dict[str, object]], report_text: str) -> None:
    lines = [
        "# Chapter 6 Taxi Python Output Summary",
        "",
        f"Run prefix: `{RUN_PREFIX}`",
        "",
        "Run report:",
        "",
        report_text.strip(),
        "",
        "Objects:",
        "",
    ]
    for item in objects:
        lines.append(f"- `{item['Key']}` ({item['Size']} bytes)")
    lines.extend(
        [
            "",
            "Generated local plots:",
            "",
            "- `taxi2_hourly_metrics.png`",
            "- `taxi3_payment_metrics.png`",
            "- `taxi4_distance_metrics.png`",
        ]
    )
    (LAB_DIR / "taxi1_output_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    require_completed(RUN_PREFIX, "Fill in RUN_PREFIX before running this script.")
    bucket = participant_bucket()
    client = s3_client()
    prefix = RUN_PREFIX.strip("/")
    objects = list_objects(client, bucket, prefix)
    if not objects:
        raise SystemExit(f"No objects found at s3://{bucket}/{prefix}")

    download_prefix(client, bucket, prefix, CACHE_DIR)
    hourly = read_parquet(CACHE_DIR / "gold_hourly_metrics.parquet")
    payment = read_parquet(CACHE_DIR / "gold_payment_metrics.parquet")
    distance = read_parquet(CACHE_DIR / "gold_distance_metrics.parquet")
    report_text = (CACHE_DIR / "taxi1_run_report.md").read_text(encoding="utf-8")

    save_hourly_plot(hourly)
    save_payment_plot(payment)
    save_distance_plot(distance)
    write_summary(objects, report_text)

    print(f"Downloaded taxi Python outputs from s3://{bucket}/{prefix}")
    print(f"Wrote {LAB_DIR / 'taxi1_output_summary.md'}")
    print(f"Wrote {LAB_DIR / 'taxi2_hourly_metrics.png'}")
    print(f"Wrote {LAB_DIR / 'taxi3_payment_metrics.png'}")
    print(f"Wrote {LAB_DIR / 'taxi4_distance_metrics.png'}")


if __name__ == "__main__":
    main()

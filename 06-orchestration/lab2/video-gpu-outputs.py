from __future__ import annotations

import os
from pathlib import Path

import boto3
import urllib3
from dotenv import load_dotenv

REPO_ROOT = Path("/root/preparation-for-ai")
LAB_DIR = REPO_ROOT / "06-orchestration" / "lab2"

# TODO:
# Fill in the run prefix printed by the video GPU Airflow DAG.
RUN_PREFIX = "TODO_FILL_VIDEO_GPU_RUN_PREFIX"
OUTPUT_FILES = {
    "video1_dataset_overview.png",
    "video2_training_curve.png",
    "video3_confusion_matrix.png",
    "video4_predictions.png",
}
STALE_OUTPUT_FILES = {"video0_output_summary.md", "video1_training_report.md"}
VIDEO_PREVIEW_EXTENSIONS = {".avi", ".mp4", ".mov", ".webm"}


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


def download_named_outputs(client, bucket: str, prefix: str) -> list[str]:
    downloaded = []
    for item in list_objects(client, bucket, prefix):
        key = item["Key"]
        filename = Path(key).name
        is_video_preview = (
            filename.startswith("video5_preview_")
            and Path(filename).suffix.lower() in VIDEO_PREVIEW_EXTENSIONS
        )
        if filename not in OUTPUT_FILES and not is_video_preview:
            continue
        local_path = LAB_DIR / filename
        client.download_file(bucket, key, str(local_path))
        downloaded.append(filename)
    return downloaded


def main() -> None:
    require_completed(RUN_PREFIX, "Fill in RUN_PREFIX before running this script.")
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES | STALE_OUTPUT_FILES:
        (LAB_DIR / filename).unlink(missing_ok=True)
    for path in LAB_DIR.glob("video5_preview_*.*"):
        path.unlink(missing_ok=True)
    bucket = participant_bucket()
    client = s3_client()
    prefix = RUN_PREFIX.strip("/")
    objects = list_objects(client, bucket, prefix)
    if not objects:
        raise SystemExit(f"No objects found at s3://{bucket}/{prefix}")

    downloaded = download_named_outputs(client, bucket, prefix)
    print(f"Downloaded outputs from s3://{bucket}/{prefix}")
    print("")
    print("Downloaded files:")
    for filename in downloaded:
        print(f"- {filename}")
    print("")
    print("All objects found:")
    for item in objects:
        print(f"- {item['Key']} ({item['Size']} bytes)")


if __name__ == "__main__":
    main()

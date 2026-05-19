from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import urllib3
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path("/root/preparation-for-ai")


def load_environment() -> None:
    load_dotenv(REPO_ROOT / "participant.env")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or "replace-me" in value or "example.invalid" in value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise SystemExit(message)


def load_s3():
    load_environment()
    endpoint_url = require_env("S3_ENDPOINT_URL")
    verify_ssl = os.environ.get("S3_VERIFY_SSL", "true").strip().lower()
    verify = verify_ssl not in {"0", "false", "no"}
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    participant_bucket = os.environ.get("PARTICIPANT_BUCKET") or require_env(
        "S3_BUCKET"
    )
    shared_bucket = os.environ.get("SHARED_BUCKET") or os.environ.get(
        "S3_SHARED_BUCKET", ""
    )
    if not shared_bucket:
        raise SystemExit("Missing SHARED_BUCKET or S3_SHARED_BUCKET in participant.env")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=require_env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("S3_SECRET_ACCESS_KEY"),
        verify=verify,
    )
    return client, participant_bucket, shared_bucket


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected an s3://bucket/key URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_object_bytes(client, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def read_manifest(client, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_object_bytes(client, bucket, key)))


def cache_s3_object(client, uri: str, cache_dir: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    cache_name = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    suffix = Path(key).suffix or ".bin"
    cache_path = cache_dir / f"{cache_name}{suffix}"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(read_object_bytes(client, bucket, key))
    return cache_path


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def write_markdown_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_class_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    counts = frame.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", figsize=(10, 5), color=plt.cm.Set2.colors)
    plt.title("Dataset examples per split and label")
    plt.xlabel("Split")
    plt.ylabel("Examples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    if "validation_loss" in history[0]:
        train_loss = [row["train_loss"] for row in history]
        validation_loss = [row["validation_loss"] for row in history]
        train_accuracy = [row["train_accuracy"] for row in history]
        validation_accuracy = [row["validation_accuracy"] for row in history]
        max_loss = max(train_loss + validation_loss)
        loss_limit = max_loss * 1.08 if max_loss > 0 else 1

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        panels = [
            (train_loss, "Training loss", "Loss", "#2563eb", (0, loss_limit)),
            (validation_loss, "Validation loss", "Loss", "#dc2626", (0, loss_limit)),
            (train_accuracy, "Training accuracy", "Accuracy", "#059669", (0, 1)),
            (
                validation_accuracy,
                "Validation accuracy",
                "Accuracy",
                "#7c3aed",
                (0, 1),
            ),
        ]
        for axis, (values, title, ylabel, color, ylim) in zip(
            axes, panels, strict=True
        ):
            axis.plot(epochs, values, marker="o", color=color)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            if ylim:
                axis.set_ylim(*ylim)
            axis.grid(alpha=0.25)
    else:
        train_loss = [row["train_loss"] for row in history]
        validation_error = [row["validation_error"] for row in history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, marker="o")
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)

        axes[1].plot(epochs, validation_error, marker="o", color="#d1495b")
        axes[1].set_title("Validation error")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    matrix,
    labels: list[str],
    output_path: Path,
    title: str = "Confusion matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = int(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.55 else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_image_preview(
    rows: list[tuple],
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError("Cannot create an image preview with zero rows.")

    columns = min(4, len(rows))
    tile_width = 230
    tile_height = 185
    image_size = 118
    padding = 12
    row_count = (len(rows) + columns - 1) // columns
    header_height = 42

    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + row_count * tile_height),
        color=(245, 244, 240),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 12), title, fill=(24, 24, 24))

    for index, row in enumerate(rows):
        path, label, note = row[:3]
        note_color = row[3] if len(row) > 3 else (82, 82, 82)
        column = index % columns
        row_number = index // columns
        x = column * tile_width
        y = header_height + row_number * tile_height
        with Image.open(path) as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
        sheet.paste(thumbnail, (x + padding, y + padding))
        draw.text((x + padding, y + padding + image_size + 8), label, fill=(24, 24, 24))
        draw.text(
            (x + padding, y + padding + image_size + 25),
            note[:32],
            fill=note_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")

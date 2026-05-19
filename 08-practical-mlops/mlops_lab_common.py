from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import torch
import urllib3
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

REPO_ROOT = Path("/root/preparation-for-ai")


def load_environment() -> None:
    load_dotenv(REPO_ROOT / "participant.env")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or "replace-me" in value or "example.invalid" in value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def require_completed(value: str, message: str) -> None:
    if str(value).startswith("TODO"):
        raise SystemExit(message)


def load_s3():
    load_environment()
    endpoint_url = require_env("S3_ENDPOINT_URL")
    verify_ssl = os.environ.get("S3_VERIFY_SSL", "true").strip().lower()
    verify = verify_ssl not in {"0", "false", "no"}
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    return client, shared_bucket


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
    cache_path = cache_dir / bucket / key
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(read_object_bytes(client, bucket, key))
    return cache_path


def build_mobilenet(output_classes: int, trainable_backbone_blocks: int):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if trainable_backbone_blocks > 0:
        for block in list(model.features.children())[-trainable_backbone_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, output_classes)
    return model, weights.transforms()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    validation_accuracy = [row["validation_accuracy"] for row in history]

    max_loss = max(train_loss + validation_loss)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(0, max_loss * 1.08)

    axes[1].plot(epochs, validation_loss, marker="o", color="#7c2d12")
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_ylim(0, max_loss * 1.08)

    axes[2].plot(epochs, train_accuracy, marker="o", color="#166534")
    axes[2].set_title("Training accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)

    axes[3].plot(epochs, validation_accuracy, marker="o", color="#2f6f9f")
    axes[3].set_title("Validation accuracy")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Accuracy")
    axes[3].set_ylim(0, 1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(matrix, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Validation confusion matrix")
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


def save_prediction_grid(
    rows: list[tuple[Path, str, str, float]],
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError("Cannot create a prediction grid with zero rows.")

    columns = min(4, len(rows))
    tile_width = 250
    tile_height = 205
    image_size = 124
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

    for index, (path, actual, predicted, confidence) in enumerate(rows):
        column = index % columns
        row_number = index // columns
        x = column * tile_width
        y = header_height + row_number * tile_height
        with Image.open(path) as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
        sheet.paste(thumbnail, (x + padding, y + padding))
        draw.text(
            (x + padding, y + padding + image_size + 8), actual, fill=(24, 24, 24)
        )
        actual_label = actual.removeprefix("actual: ")
        predicted_label = predicted.removeprefix("predicted: ")
        prediction_color = (
            (22, 101, 52) if actual_label == predicted_label else (185, 28, 28)
        )
        draw.text(
            (x + padding, y + padding + image_size + 25),
            f"{predicted} ({confidence:.2f})"[:34],
            fill=prediction_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")

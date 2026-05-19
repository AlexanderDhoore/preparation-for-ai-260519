from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import av
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-validation-error", type=float, default=0.90)
    return parser.parse_args()


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://s3.mechatronics.lan"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def read_manifest(client, bucket: str, key: str) -> pd.DataFrame:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.csv"
        client.download_file(bucket, key, str(path))
        return pd.read_csv(path)


def cache_clip(client, uri: str, cache_dir: Path) -> Path:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3 URI, got {uri}")
    without_scheme = uri.removeprefix("s3://")
    bucket, key = without_scheme.split("/", 1)
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}{Path(key).suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(path))
    return path


def read_video_frames(path: Path) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        for frame in container.decode(video_stream):
            image = frame.to_image().convert("RGB")
            frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"Video clip has no decoded frames: {path}")
    return torch.stack(frames)


class VideoDataset(Dataset):
    def __init__(self, frame, client, cache_dir: Path, transform, label_to_index):
        self.frame = frame.reset_index(drop=True)
        self.client = client
        self.cache_dir = cache_dir
        self.transform = transform
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        clip_path = cache_clip(self.client, row.asset_uri, self.cache_dir)
        video = read_video_frames(clip_path)
        if len(video) < 8:
            raise ValueError(f"Video clip is too short: {row.asset_uri}")
        indices = torch.linspace(0, len(video) - 1, steps=16).long()
        frames = video[indices].float() / 255.0
        frames = self.transform(frames)
        label = torch.tensor(self.label_to_index[row.label_name], dtype=torch.long)
        return frames, label


def build_model(label_count: int, device: torch.device):
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, label_count)
    return model.to(device), weights.transforms()


def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(videos)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for videos, labels in loader:
            output = model(videos.to(device)).argmax(dim=1).cpu()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(output.tolist())
    return true_labels, predicted_labels, accuracy_score(true_labels, predicted_labels)


def save_training_curve(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_title("Training loss")
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        marker="o",
        color="#2f6f9f",
    )
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[2].plot(
        epochs,
        [row["validation_error"] for row in history],
        marker="o",
        color="#c43b3b",
    )
    axes[2].set_title("Validation error")
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_dataset_overview(manifest: pd.DataFrame, path: Path) -> None:
    counts = (
        manifest.groupby(["label_name", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    sizes = (
        manifest.assign(megabytes=manifest["bytes"] / (1024 * 1024))
        .groupby("label_name")["megabytes"]
        .mean()
        .reindex(counts.index)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts.plot(kind="bar", ax=axes[0], color=["#2f6f9f", "#d1495b"])
    axes[0].set_title("Video clips per action class")
    axes[0].set_xlabel("Action class")
    axes[0].set_ylabel("Clip count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Split")

    axes[1].bar(sizes.index, sizes.values, color="#6f7f2f")
    axes[1].set_title("Average clip size")
    axes[1].set_xlabel("Action class")
    axes[1].set_ylabel("Megabytes")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Shared video-action dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion(matrix, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("True action")
    ax.set_title("Video action confusion matrix")
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_prediction_grid(
    client,
    validation_frame: pd.DataFrame,
    predicted_labels: list[int],
    index_to_label: dict[int, str],
    cache_dir: Path,
    path: Path,
) -> None:
    indexed = validation_frame.reset_index(drop=True)
    sample = indexed.groupby("label_name", sort=True).head(4)
    rows = []
    for row in sample.itertuples():
        clip_path = cache_clip(client, row.asset_uri, cache_dir)
        video = read_video_frames(clip_path)
        frames = []
        for frame_index in [0, len(video) // 2, len(video) - 1]:
            frame = video[frame_index].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame).convert("RGB"))
        prediction = index_to_label[predicted_labels[row.Index]]
        rows.append((frames, row.label_name, prediction))

    columns = min(4, len(rows))
    tile_width = 330
    tile_height = 175
    frame_size = 82
    padding = 12
    row_count = (len(rows) + columns - 1) // columns
    header_height = 42
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + row_count * tile_height),
        color=(245, 244, 240),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (padding, 12),
        "Video action predictions: first, middle, and last frame",
        fill=(24, 24, 24),
    )

    for index, (frames, actual, prediction) in enumerate(rows):
        column = index % columns
        row_number = index // columns
        x = column * tile_width
        y = header_height + row_number * tile_height
        for frame_offset, image in enumerate(frames):
            thumbnail = ImageOps.fit(image, (frame_size, frame_size))
            sheet.paste(
                thumbnail,
                (x + padding + frame_offset * (frame_size + 6), y + padding),
            )
        color = (22, 101, 52) if actual == prediction else (185, 28, 28)
        draw.text(
            (x + padding, y + padding + frame_size + 8),
            f"actual: {actual}",
            fill=(24, 24, 24),
        )
        draw.text(
            (x + padding, y + padding + frame_size + 25),
            f"predicted: {prediction}",
            fill=color,
        )

    sheet.save(path, format="PNG")


def safe_label_name(label: str) -> str:
    cleaned = [
        character.lower() if character.isalnum() else "-" for character in label
    ]
    return "".join(cleaned).strip("-")


def save_video_previews(
    client,
    validation_frame: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    sample = validation_frame.sort_values("label_name").groupby("label_name").head(1)
    for row in sample.itertuples():
        clip_path = cache_clip(client, row.asset_uri, cache_dir)
        preview_name = (
            f"video5_preview_{safe_label_name(row.label_name)}{clip_path.suffix}"
        )
        shutil.copyfile(clip_path, output_dir / preview_name)


def upload_file(client, bucket: str, prefix: str, path: Path) -> None:
    client.upload_file(str(path), bucket, f"{prefix.rstrip('/')}/{path.name}")


def main() -> None:
    args = parse_args()
    client = s3_client()
    manifest = read_manifest(client, args.shared_bucket, args.manifest_key)
    labels = sorted(manifest["label_name"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training job is expected to run on a GPU pod.")

    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        model, transform = build_model(len(labels), device)
        train_loader = DataLoader(
            VideoDataset(
                train_frame, client, work_dir / "clips", transform, label_to_index
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            VideoDataset(
                validation_frame,
                client,
                work_dir / "clips",
                transform,
                label_to_index,
            ),
            batch_size=args.batch_size,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)
        loss_function = nn.CrossEntropyLoss()
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_function, device
            )
            true_labels, predicted_labels, accuracy = evaluate(
                model, validation_loader, device
            )
            validation_error = 1.0 - accuracy
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_accuracy": accuracy,
                    "validation_error": validation_error,
                }
            )
            print(
                f"Epoch {epoch}: training loss {train_loss:.3f}, "
                f"validation accuracy {accuracy:.3f}, "
                f"validation error {validation_error:.3f}"
            )

        output_dir = work_dir / "outputs"
        output_dir.mkdir()
        save_dataset_overview(manifest, output_dir / "video1_dataset_overview.png")
        save_training_curve(history, output_dir / "video2_training_curve.png")
        matrix = confusion_matrix(
            true_labels, predicted_labels, labels=range(len(labels))
        )
        save_confusion(matrix, labels, output_dir / "video3_confusion_matrix.png")
        save_prediction_grid(
            client,
            validation_frame,
            predicted_labels,
            index_to_label,
            work_dir / "clips",
            output_dir / "video4_predictions.png",
        )
        save_video_previews(
            client,
            validation_frame,
            work_dir / "clips",
            output_dir,
        )

        final_error = history[-1]["validation_error"]
        print(f"Labels: {', '.join(labels)}")
        print(f"Training examples: {len(train_frame)}")
        print(f"Validation examples: {len(validation_frame)}")
        print(f"Final validation error: {final_error:.3f}")
        print(f"Required maximum validation error: {args.max_validation_error:.3f}")

        for path in output_dir.iterdir():
            upload_file(client, args.output_bucket, args.output_prefix, path)

        if final_error > args.max_validation_error:
            raise RuntimeError(
                f"Validation failed: error {final_error:.3f} is above "
                f"max_validation_error={args.max_validation_error:.3f}."
            )


if __name__ == "__main__":
    main()

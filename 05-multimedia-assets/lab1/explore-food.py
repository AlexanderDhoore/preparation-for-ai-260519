from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/05-multimedia-assets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multimedia_lab_common import (
    cache_s3_object,
    load_s3,
    read_manifest,
    require_columns,
)

LAB_DIR = Path("/root/preparation-for-ai/05-multimedia-assets/lab1")
CACHE_DIR = LAB_DIR / "cache"

MANIFEST_KEY = "datasets/chapter05/food101/manifest.csv"


def save_manifest_preview(manifest) -> Path:
    path = LAB_DIR / "explore1_manifest_preview.csv"
    columns = [
        "asset_uri",
        "label_name",
        "split",
        "width",
        "height",
        "sha256",
    ]
    preview = (
        manifest.sort_values(["label_name", "split", "asset_uri"])
        .groupby("label_name", sort=True)
        .head(1)
        .head(20)
    )
    preview[columns].to_csv(path, index=False)
    return path


def save_overview(manifest) -> Path:
    path = LAB_DIR / "explore2_dataset_overview.png"
    split_counts = manifest["split"].value_counts().sort_index()
    label_counts = manifest.groupby("label_name").size()
    size_sample = manifest.sample(n=min(7000, len(manifest)), random_state=42)
    orientation_counts = {
        "landscape": int((manifest["width"] > manifest["height"]).sum()),
        "square": int((manifest["width"] == manifest["height"]).sum()),
        "portrait": int((manifest["width"] < manifest["height"]).sum()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    axes[0].bar(split_counts.index, split_counts.values, color="#2563eb")
    axes[0].set_title("Images Per Split")
    axes[0].set_ylabel("Images")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        range(len(label_counts)), label_counts.sort_values().values, color="#059669"
    )
    axes[1].set_title("Images Per Class")
    axes[1].set_xlabel("Food classes")
    axes[1].set_ylabel("Images")
    axes[1].set_ylim(0, max(label_counts.max() * 1.08, 1))
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].scatter(
        size_sample["height"],
        size_sample["width"],
        s=10,
        alpha=0.24,
        color="#7c3aed",
        linewidths=0,
    )
    axes[2].set_title("Image Size Distribution")
    axes[2].set_xlabel("Height in pixels")
    axes[2].set_ylabel("Width in pixels")
    axes[2].grid(alpha=0.2)

    axes[3].bar(
        orientation_counts.keys(),
        orientation_counts.values(),
        color=["#0f766e", "#ca8a04", "#be123c"],
    )
    axes[3].set_title("Image Orientation")
    axes[3].set_ylabel("Images")
    axes[3].grid(axis="y", alpha=0.2)

    fig.suptitle(f"Food-101 Shared Manifest: {len(manifest):,} Images")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_examples(client, manifest) -> Path:
    path = LAB_DIR / "explore3_label_examples.png"
    labels = sorted(manifest["label_name"].unique())
    columns = 6
    rows = (len(labels) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(18, 32))
    axes = axes.flatten()
    for axis, label in zip(axes, labels, strict=False):
        row = manifest.loc[manifest["label_name"] == label].iloc[0]
        image_path = cache_s3_object(client, row.asset_uri, CACHE_DIR / "images")
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_title(label, fontsize=8)
        axis.axis("off")
    for axis in axes[len(labels) :]:
        axis.axis("off")
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    client, _participant_bucket, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, MANIFEST_KEY)
    require_columns(
        manifest,
        {"asset_uri", "label_name", "split", "width", "height", "sha256"},
        "Food-101 manifest",
    )

    preview = save_manifest_preview(manifest)
    overview = save_overview(manifest)
    examples = save_examples(client, manifest)

    print(f"Read manifest: s3://{shared_bucket}/{MANIFEST_KEY}")
    print(f"Images: {len(manifest):,}")
    print(f"Labels: {manifest['label_name'].nunique()}")
    print(f"Wrote preview: {preview}")
    print(f"Wrote plot: {overview}")
    print(f"Wrote plot: {examples}")


if __name__ == "__main__":
    main()

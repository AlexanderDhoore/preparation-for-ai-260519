from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/05-multimedia-assets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multimedia_lab_common import (
    load_s3,
    read_manifest,
    require_columns,
)

LAB_DIR = Path("/root/preparation-for-ai/05-multimedia-assets/lab2")

MANIFEST_KEY = "datasets/chapter05/banking77/manifest.csv"


def save_manifest_preview(manifest) -> Path:
    path = LAB_DIR / "explore1_manifest_preview.csv"
    columns = [
        "asset_uri",
        "label_name",
        "split",
        "characters",
        "sha256",
        "text_preview",
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
    manifest = manifest.copy()
    manifest["words"] = manifest["text_preview"].str.split().str.len()
    split_counts = manifest["split"].value_counts().sort_index()
    label_counts = manifest.groupby("label_name").size().sort_values()
    split_by_label = (
        manifest.groupby(["label_name", "split"]).size().unstack(fill_value=0)
    )
    longest_labels = (
        manifest.groupby("label_name")["words"].median().sort_values().tail(12)
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    axes[0].bar(split_counts.index, split_counts.values, color="#2563eb")
    axes[0].set_title("Rows Per Split")
    axes[0].set_ylabel("Texts")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(range(len(label_counts)), label_counts.values, color="#059669")
    axes[1].set_title("Texts Per Intent Label")
    axes[1].set_xlabel("Intent labels, sorted by size")
    axes[1].set_ylabel("Texts")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].hist(manifest["words"], bins=24, color="#7c3aed")
    axes[2].set_title("Word Counts")
    axes[2].set_xlabel("Approximate words")
    axes[2].set_ylabel("Texts")
    axes[2].grid(axis="y", alpha=0.2)

    split_by_label = split_by_label.reindex(label_counts.index)
    x_positions = range(len(split_by_label))
    train_values = split_by_label.get("train", 0).values
    validation_values = split_by_label.get("validation", 0).values
    axes[3].bar(x_positions, train_values, label="train", color="#2563eb")
    axes[3].bar(
        x_positions,
        validation_values,
        bottom=train_values,
        label="validation",
        color="#f59e0b",
    )
    axes[3].set_title("Split Balance Per Intent")
    axes[3].set_xlabel("Intent labels, sorted by total texts")
    axes[3].set_ylabel("Texts")
    axes[3].legend()
    axes[3].grid(axis="y", alpha=0.2)

    axes[4].barh(longest_labels.index, longest_labels.values, color="#0f766e")
    axes[4].set_title("Highest Median Word Count Intents")
    axes[4].set_xlabel("Median approximate words")
    axes[4].grid(axis="x", alpha=0.2)

    boxplot_data = [
        manifest.loc[manifest["split"] == split, "words"]
        for split in split_counts.index
    ]
    axes[5].boxplot(boxplot_data, tick_labels=split_counts.index, showfliers=False)
    axes[5].set_title("Word Counts By Split")
    axes[5].set_ylabel("Approximate words")
    axes[5].grid(axis="y", alpha=0.2)

    fig.suptitle(f"Banking77 Shared Manifest: {len(manifest):,} Texts")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_label_menu(client, manifest) -> Path:
    path = LAB_DIR / "explore3_label_menu.md"
    labels = sorted(manifest["label_name"].unique())
    lines = [
        "# Banking77 Label Menu",
        "",
        "Pick five intent labels for the transformer lab. Each section shows one",
        "short validation example and the available train/validation counts.",
        "",
    ]
    for label in labels:
        label_frame = manifest.loc[manifest["label_name"] == label]
        counts = label_frame["split"].value_counts()
        train_count = int(counts.get("train", 0))
        validation_count = int(counts.get("validation", 0))
        example_frame = label_frame.loc[label_frame["split"] == "validation"]
        if example_frame.empty:
            example_frame = label_frame
        row = example_frame.sort_values("characters").iloc[len(example_frame) // 2]
        text = str(row.text_preview)
        lines.extend(
            [
                f"## {label}",
                "",
                "\n".join(textwrap.wrap(text, width=90)),
                "",
                f"train {train_count} / validation {validation_count}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    client, _participant_bucket, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, MANIFEST_KEY)
    require_columns(
        manifest,
        {
            "asset_uri",
            "label_name",
            "split",
            "characters",
            "sha256",
            "text_preview",
        },
        "Banking77 manifest",
    )

    preview = save_manifest_preview(manifest)
    overview = save_overview(manifest)
    label_menu = write_label_menu(client, manifest)

    print(f"Read manifest: s3://{shared_bucket}/{MANIFEST_KEY}")
    print(f"Texts: {len(manifest):,}")
    print(f"Intent labels: {manifest['label_name'].nunique()}")
    print(f"Wrote preview: {preview}")
    print(f"Wrote plot: {overview}")
    print(f"Wrote label menu: {label_menu}")


if __name__ == "__main__":
    main()

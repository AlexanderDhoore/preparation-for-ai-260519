from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/02-object-storage")

import matplotlib.pyplot as plt
import pandas as pd

from s3_lab_common import load_s3, read_manifest, read_object_bytes, read_wav_bytes

OUTPUT_DIR = Path("/root/preparation-for-ai/02-object-storage/lab1")
BRONZE_MANIFEST_KEY = "bronze/esc50/manifest.csv"
SAMPLE_LABELS = [
    "dog",
    "rain",
    "chirping_birds",
    "clock_alarm",
    "chainsaw",
    "breathing",
]


def safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def audio_metadata(content: bytes) -> dict[str, float | int]:
    samples, sample_rate, channels, bits = read_wav_bytes(content)
    return {
        "duration_s": round(float(len(samples) / sample_rate), 3),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bits_per_sample": bits,
    }


def write_class_counts(manifest: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore2_class_counts.png"
    counts = manifest["label_text"].value_counts().sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(10, 12))
    axis.barh(counts.index, counts.values, color="#2563eb")
    axis.set_title("Bronze ESC-50 Class Counts")
    axis.set_xlabel("Audio clips")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_metadata_plot(metadata: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore3_audio_metadata.png"
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    duration_counts = metadata["duration_s"].round(2).value_counts().sort_index()
    axes[0].bar(
        [f"{value:.2f}" for value in duration_counts.index],
        duration_counts.values,
        color="#4c956c",
    )
    axes[0].set_title("Duration")
    axes[0].set_xlabel("Seconds")

    sample_rate_counts = metadata["sample_rate_hz"].value_counts().sort_index()
    axes[1].bar(
        [f"{int(value)}" for value in sample_rate_counts.index],
        sample_rate_counts.values,
        color="#f59e0b",
    )
    axes[1].set_title("Sample Rate")
    axes[1].set_xlabel("Hz")

    size_counts = (metadata["bytes"] / 1024).round(1).value_counts().sort_index()
    axes[2].bar(
        [f"{value:.1f}" for value in size_counts.index],
        size_counts.values,
        color="#d7263d",
    )
    axes[2].set_title("Object Size")
    axes[2].set_xlabel("KiB")
    for axis in axes:
        axis.set_ylabel("Clips")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Bronze Audio Metadata")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_waveform_samples(samples: list[dict[str, object]]) -> Path:
    path = OUTPUT_DIR / "explore4_waveform_samples.png"
    figure, axes = plt.subplots(len(samples), 1, figsize=(11, 10), sharex=True)
    for axis, sample in zip(axes, samples, strict=True):
        values = sample["samples"]
        sample_rate = sample["sample_rate_hz"]
        duration = len(values) / sample_rate
        step = max(1, len(values) // 6000)
        x_values = [index / sample_rate for index in range(0, len(values), step)]
        axis.plot(x_values, values[::step], linewidth=0.7, color="#2563eb")
        axis.set_title(str(sample["label_text"]), loc="left")
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlim(0, duration)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Seconds")
    figure.suptitle("Example Bronze Waveforms")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client, _participant_bucket, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, BRONZE_MANIFEST_KEY)
    manifest_path = OUTPUT_DIR / "explore1_bronze_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    samples: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for label in SAMPLE_LABELS:
        row = manifest[manifest["label_text"] == label].sort_values("key").iloc[0]
        content = read_object_bytes(client, shared_bucket, row.key)
        values, sample_rate, channels, bits = read_wav_bytes(content)
        local_path = OUTPUT_DIR / f"explore_sample_{safe_name(label)}.wav"
        local_path.write_bytes(content)
        samples.append(
            {
                "label_text": label,
                "samples": values,
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "bits_per_sample": bits,
            }
        )

    metadata_sample = (
        manifest.sort_values(["label_text", "key"])
        .groupby("label_text", group_keys=False)
        .head(2)
    )
    for row in metadata_sample.itertuples(index=False):
        content = read_object_bytes(client, shared_bucket, row.key)
        metadata_rows.append(
            {
                "label_text": row.label_text,
                "fold": int(row.fold),
                "key": row.key,
                "bytes": int(row.bytes),
                **audio_metadata(content),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata_path = OUTPUT_DIR / "explore_audio_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    outputs = [
        manifest_path,
        write_class_counts(manifest),
        write_metadata_plot(metadata),
        write_waveform_samples(samples),
        metadata_path,
    ]
    for output in outputs:
        print(f"Wrote {output}")
    for label in SAMPLE_LABELS:
        print(f"Wrote {OUTPUT_DIR / f'explore_sample_{safe_name(label)}.wav'}")


if __name__ == "__main__":
    main()

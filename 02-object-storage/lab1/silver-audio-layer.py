from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/02-object-storage")

import matplotlib.pyplot as plt
import pandas as pd

from s3_lab_common import (
    load_s3,
    read_manifest,
    read_object_bytes,
    sha256_bytes,
    upload_bytes,
    upload_csv,
    upload_file,
)

OUTPUT_DIR = Path("/root/preparation-for-ai/02-object-storage/lab1")
BRONZE_MANIFEST_KEY = "bronze/esc50/manifest.csv"
SILVER_MANIFEST_KEY = "silver/chapter02/esc50/audio_manifest.csv"

# TODO 1:
# Choose exactly five ESC-50 labels from explore2_class_counts.png.
# Use the exact label text, for example: "dog", "rain", "clock_alarm".
SELECTED_LABELS = [
    "TODO_CHOOSE_LABEL_1",
    "TODO_CHOOSE_LABEL_2",
    "TODO_CHOOSE_LABEL_3",
    "TODO_CHOOSE_LABEL_4",
    "TODO_CHOOSE_LABEL_5",
]


def safe_label(label: str) -> str:
    return label.lower().replace("/", "-").replace(" ", "_")


def require_labels(labels: list[str], available: set[str]) -> None:
    if len(labels) != 5 or any(label.startswith("TODO") for label in labels):
        raise NotImplementedError("TODO 1: choose exactly five ESC-50 labels.")
    unknown = sorted(set(labels) - available)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")


def write_selected_counts(silver: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "silver1_selected_class_counts.png"
    counts = silver["label_text"].value_counts().sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(counts.index, counts.values, color="#4c956c")
    axis.set_title("Silver Class Counts")
    axis.set_xlabel("Audio clips copied into your bucket")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client, participant_bucket, shared_bucket = load_s3()
    bronze = read_manifest(client, shared_bucket, BRONZE_MANIFEST_KEY)
    selected_labels = list(dict.fromkeys(SELECTED_LABELS))
    require_labels(selected_labels, set(bronze["label_text"]))

    selected = bronze[bronze["label_text"].isin(selected_labels)].copy()
    rows: list[dict[str, object]] = []
    for row in selected.sort_values(["label_text", "key"]).itertuples(index=False):
        content = read_object_bytes(client, shared_bucket, row.key)
        number = Path(row.key).stem
        silver_key = (
            f"silver/chapter02/esc50/audio/{safe_label(row.label_text)}/{number}.wav"
        )
        upload_bytes(client, participant_bucket, silver_key, content, "audio/wav")
        print(f"Copied s3://{shared_bucket}/{row.key}")
        print(f"    -> s3://{participant_bucket}/{silver_key}")
        rows.append(
            {
                "dataset": row.dataset,
                "layer": "silver",
                "label_id": int(row.label_id),
                "label_text": row.label_text,
                "fold": int(row.fold),
                "bucket": participant_bucket,
                "key": silver_key,
                "source_uri": f"s3://{shared_bucket}/{row.key}",
                "source_sha256": row.sha256,
                "sha256": sha256_bytes(content),
                "content_type": "audio/wav",
                "bytes": len(content),
            }
        )

    silver = pd.DataFrame(rows)
    local_manifest = OUTPUT_DIR / "silver_audio_manifest.csv"
    silver.to_csv(local_manifest, index=False)
    plot_path = write_selected_counts(silver)

    upload_csv(client, participant_bucket, SILVER_MANIFEST_KEY, silver)
    upload_file(
        client,
        participant_bucket,
        "outputs/chapter02/lab1/silver1_selected_class_counts.png",
        plot_path,
        "image/png",
    )

    print(f"Wrote {local_manifest}")
    print(f"Wrote {plot_path}")
    print(
        f"Uploaded Silver manifest to s3://{participant_bucket}/{SILVER_MANIFEST_KEY}"
    )


if __name__ == "__main__":
    main()

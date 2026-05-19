from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/02-object-storage")

import matplotlib.pyplot as plt
import pandas as pd

from s3_lab_common import (
    audio_features,
    load_s3,
    read_object_bytes,
    upload_csv,
    upload_file,
)

OUTPUT_DIR = Path("/root/preparation-for-ai/02-object-storage/lab1")
SILVER_MANIFEST_KEY = "silver/chapter02/esc50/audio_manifest.csv"
GOLD_FEATURES_KEY = "gold/chapter02/esc50/audio_features.csv"
FEATURE_COLUMNS = [
    "rms_energy",
    "peak_amplitude",
    "zero_crossing_rate",
    "mean_absolute_amplitude",
    "spectral_centroid",
    "very_low_frequency_share",
    "low_frequency_share",
    "mid_frequency_share",
    "high_frequency_share",
    "very_high_frequency_share",
]


def read_csv_from_s3(client, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_object_bytes(client, bucket, key)))


def write_feature_overview(gold: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "gold1_feature_overview.png"
    figure, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()
    labels = sorted(gold["label_text"].unique())
    for axis, column in zip(axes, FEATURE_COLUMNS, strict=True):
        data = [gold.loc[gold["label_text"] == label, column] for label in labels]
        axis.boxplot(data, tick_labels=labels, vert=True)
        axis.set_title(column)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Gold Audio Features By Label")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_correlation(gold: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "gold2_feature_correlation.png"
    correlation = gold[FEATURE_COLUMNS].corr().fillna(0)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(
        range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=65, ha="right"
    )
    axis.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    axis.set_title("Gold Feature Correlation")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client, participant_bucket, _shared_bucket = load_s3()
    silver = read_csv_from_s3(client, participant_bucket, SILVER_MANIFEST_KEY)

    rows: list[dict[str, object]] = []
    for row in silver.sort_values(["label_text", "key"]).itertuples(index=False):
        content = read_object_bytes(client, participant_bucket, row.key)
        features = audio_features(content)
        rows.append(
            {
                "label_id": int(row.label_id),
                "label_text": row.label_text,
                "fold": int(row.fold),
                "source_key": row.key,
                **{
                    key: round(float(value), 8)
                    for key, value in features.items()
                    if isinstance(value, float)
                },
                **{
                    key: int(value)
                    for key, value in features.items()
                    if isinstance(value, int)
                },
            }
        )

    gold = pd.DataFrame(rows)
    local_features = OUTPUT_DIR / "gold_audio_features.csv"
    gold.to_csv(local_features, index=False)
    plots = [
        write_feature_overview(gold),
        write_correlation(gold),
    ]

    upload_csv(client, participant_bucket, GOLD_FEATURES_KEY, gold)
    for plot in plots:
        upload_file(
            client,
            participant_bucket,
            f"outputs/chapter02/lab1/{plot.name}",
            plot,
            "image/png",
        )

    print(f"Wrote {local_features}")
    for plot in plots:
        print(f"Wrote {plot}")
    print(f"Uploaded Gold features to s3://{participant_bucket}/{GOLD_FEATURES_KEY}")


if __name__ == "__main__":
    main()

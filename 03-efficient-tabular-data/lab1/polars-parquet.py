from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, "/root/preparation-for-ai/03-efficient-tabular-data")

from s3_tabular_common import (  # noqa: E402
    delete_prefix,
    list_keys,
    participant_bucket,
    polars_storage_options,
    s3_client,
    s3_uri,
    shared_bucket,
)

OUTPUT_DIR = Path("/root/preparation-for-ai/03-efficient-tabular-data/lab1")
TARGET_COLUMN = "cover_type"
SILVER_PREFIX = "silver/chapter03/covertype"

TARGET_LABELS = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}

# TODO 1:
# Fill in the shared Bronze CSV key from the README.
# This is the exchange-style file that you will convert to Parquet.
SOURCE_CSV_KEY = "TODO_FILL_SHARED_BRONZE_CSV_KEY"

# TODO 2:
# Choose the column used to partition your Silver Parquet output.
# Hint: pick the column with four broad land-area categories.
PARTITION_COLUMN = "TODO_FILL_PARTITION_COLUMN"


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise NotImplementedError(message)


def target_label(value: int) -> str:
    return TARGET_LABELS.get(value, str(value))


def object_size(client, bucket: str, key: str) -> int:
    return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])


def total_prefix_size(client, bucket: str, prefix: str) -> int:
    total = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        total += sum(int(item["Size"]) for item in page.get("Contents", []))
    return total


def bytes_to_mb(value: int) -> float:
    return round(value / 1024 / 1024, 2)


def build_silver_query(
    source_uri: str, storage_options: dict[str, str]
) -> pl.LazyFrame:
    return pl.scan_csv(
        source_uri,
        storage_options=storage_options,
        infer_schema_length=10000,
    ).with_columns(
        pl.col("wilderness_area").cast(pl.Int16),
        pl.col("soil_type").cast(pl.Int16),
        pl.col(TARGET_COLUMN).cast(pl.Int16),
    )


def class_balance_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(["wilderness_area", TARGET_COLUMN])
        .len()
        .rename({"len": "rows"})
        .sort(["wilderness_area", TARGET_COLUMN])
    )


def write_class_balance(summary: pl.DataFrame, by_partition: pl.DataFrame) -> Path:
    path = OUTPUT_DIR / "polars3_class_balance.png"
    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
    labels = [f"{value}: {target_label(value)}" for value in TARGET_LABELS]

    panels: list[tuple[str, pl.DataFrame]] = [("All wilderness areas", summary)]
    panels.extend(
        (
            f"Wilderness area {area}",
            by_partition.filter(pl.col("wilderness_area") == area),
        )
        for area in sorted(by_partition["wilderness_area"].unique().to_list())
    )

    for axis, (title, panel) in zip(axes, panels, strict=True):
        row_counts = {
            int(row[TARGET_COLUMN]): int(row["rows"]) for row in panel.to_dicts()
        }
        values = [row_counts.get(value, 0) for value in TARGET_LABELS]
        axis.bar(labels, values, color="#2563eb")
        axis.set_title(title)
        axis.set_ylabel("Rows")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Cover type")
    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle("Cover Type Balance Overall And Within Each Partition", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_partition_balance(summary: pl.DataFrame) -> Path:
    path = OUTPUT_DIR / "polars2_rows_by_wilderness_area.png"
    labels = [str(value) for value in summary["wilderness_area"].to_list()]
    rows = summary["rows"].to_list()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, rows, color="#059669")
    ax.set_title("Rows per wilderness area partition")
    ax.set_xlabel("Wilderness area")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_storage_summary(
    csv_bytes: int,
    parquet_bytes: int,
    written_objects: int,
) -> Path:
    path = OUTPUT_DIR / "polars1_storage_summary.png"
    csv_mb = bytes_to_mb(csv_bytes)
    parquet_mb = bytes_to_mb(parquet_bytes)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(
        ["Bronze CSV", "Silver Parquet"],
        [csv_mb, parquet_mb],
        color=["#9ca3af", "#2563eb"],
    )
    ax.set_title("Storage Size After Conversion")
    ax.set_ylabel("Megabytes")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, [csv_mb, parquet_mb], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f} MB",
            ha="center",
            va="bottom",
        )
    ax.text(
        0.5,
        0.92,
        f"Silver Parquet objects written: {written_objects}\n"
        f"Partition column: {PARTITION_COLUMN}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    require_completed(SOURCE_CSV_KEY, "TODO 1: fill in SOURCE_CSV_KEY.")
    require_completed(PARTITION_COLUMN, "TODO 2: fill in PARTITION_COLUMN.")

    client = s3_client()
    source_bucket = shared_bucket()
    target_bucket = participant_bucket()
    storage_options = polars_storage_options()

    source_uri = s3_uri(source_bucket, SOURCE_CSV_KEY)
    silver_uri = s3_uri(target_bucket, SILVER_PREFIX)
    silver = build_silver_query(source_uri, storage_options)

    available_columns = silver.collect_schema().names()
    if PARTITION_COLUMN not in available_columns:
        raise SystemExit(
            f"Partition column {PARTITION_COLUMN!r} is not available. "
            f"Available columns: {available_columns}"
        )

    class_summary = (
        silver.group_by(TARGET_COLUMN)
        .agg(
            pl.len().alias("rows"),
            pl.col("elevation").mean().round(1).alias("avg_elevation"),
        )
        .sort(TARGET_COLUMN)
        .collect()
    )
    partition_summary = (
        silver.group_by(PARTITION_COLUMN)
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).n_unique().alias("cover_type_count"),
            pl.col("elevation").mean().round(1).alias("avg_elevation"),
        )
        .sort(PARTITION_COLUMN)
        .collect()
    )

    delete_prefix(client, target_bucket, SILVER_PREFIX)
    frame = silver.collect()
    class_by_partition = class_balance_frame(frame)
    frame.write_parquet(
        silver_uri,
        partition_by=PARTITION_COLUMN,
        compression="zstd",
        statistics=True,
        storage_options=storage_options,
    )

    written_keys = list_keys(client, target_bucket, SILVER_PREFIX)
    csv_bytes = object_size(client, source_bucket, SOURCE_CSV_KEY)
    parquet_bytes = total_prefix_size(client, target_bucket, SILVER_PREFIX)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = [
        write_storage_summary(csv_bytes, parquet_bytes, len(written_keys)),
        write_partition_balance(partition_summary),
        write_class_balance(class_summary, class_by_partition),
    ]

    for plot in plots:
        print(f"Wrote plot: {plot}")
    print(f"Wrote Silver Parquet: {silver_uri}")
    print(f"CSV size: {bytes_to_mb(csv_bytes)} MB")
    print(f"Parquet size: {bytes_to_mb(parquet_bytes)} MB")
    print(f"Parquet objects written: {len(written_keys)}")
    print(class_summary)
    print(partition_summary)


if __name__ == "__main__":
    main()

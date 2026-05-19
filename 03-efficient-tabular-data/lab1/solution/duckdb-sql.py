from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/preparation-for-ai/03-efficient-tabular-data")

from s3_tabular_common import (  # noqa: E402
    configure_duckdb_s3,
    delete_prefix,
    participant_bucket,
    s3_client,
    s3_uri,
)

OUTPUT_DIR = Path("/root/preparation-for-ai/03-efficient-tabular-data/lab1/solution")

SILVER_PREFIX = "silver/chapter03/covertype"
GOLD_METRICS_KEY = "gold/chapter03/duckdb_metrics.parquet"
GROUP_COLUMN = "cover_type"
MEASUREMENT_COLUMN = "elevation"
TARGET_LABELS = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}


def require_completed(value: str, message: str) -> None:
    if value.startswith("TODO"):
        raise NotImplementedError(message)


def safe_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise SystemExit(f"Unsafe SQL identifier: {value!r}")
    return value


def plot_label(value: object) -> str:
    if GROUP_COLUMN == "cover_type" and value in TARGET_LABELS:
        return f"{value}: {TARGET_LABELS[value]}"
    return str(value)


def write_metric_plot(metrics) -> Path:
    path = OUTPUT_DIR / "duckdb2_metric_graphs.png"
    labels = [plot_label(value) for value in metrics[GROUP_COLUMN].tolist()]
    metric_columns = [column for column in metrics.columns if column != GROUP_COLUMN]
    colors = ["#2563eb", "#0891b2", "#7c3aed", "#059669", "#ca8a04", "#dc2626"]

    fig, axes = plt.subplots(
        len(metric_columns),
        1,
        figsize=(11, 2.8 * len(metric_columns)),
        sharey=True,
    )
    if len(metric_columns) == 1:
        axes = [axes]

    for index, (axis, column) in enumerate(zip(axes, metric_columns, strict=True)):
        axis.barh(labels, metrics[column].tolist(), color=colors[index % len(colors)])
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel(column)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()

    fig.suptitle("DuckDB Gold Metrics By Cover Type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_metric_table(metrics) -> Path:
    path = OUTPUT_DIR / "duckdb1_metric_table.csv"
    metrics.to_csv(path, index=False)
    return path


def main() -> None:
    require_completed(SILVER_PREFIX, "TODO 1: fill in SILVER_PREFIX.")

    group_column = safe_identifier(GROUP_COLUMN)
    measurement_column = safe_identifier(MEASUREMENT_COLUMN)

    bucket = participant_bucket()
    client = s3_client()
    feature_glob = s3_uri(bucket, f"{SILVER_PREFIX}/**/*.parquet")
    metrics_uri = s3_uri(bucket, GOLD_METRICS_KEY)

    con = duckdb.connect()
    configure_duckdb_s3(con)

    available_columns = {
        row[0]
        for row in con.sql(
            f"describe select * from read_parquet('{feature_glob}', "
            "hive_partitioning=true)"
        ).fetchall()
    }
    for column in [group_column, measurement_column]:
        if column not in available_columns:
            raise SystemExit(
                f"Column {column!r} is not available. "
                f"Available columns: {sorted(available_columns)}"
            )

    summary_sql = f"""
    select
        {group_column},
        count(*) as rows,
        count(distinct wilderness_area) as wilderness_areas,
        count(distinct soil_type) as soil_types,
        round(avg({measurement_column}), 2) as avg_{measurement_column},
        round(min({measurement_column}), 2) as min_{measurement_column},
        round(max({measurement_column}), 2) as max_{measurement_column}
    from read_parquet('{feature_glob}', hive_partitioning=true)
    group by {group_column}
    order by {group_column}
    """
    assert (
        "TODO" not in summary_sql
    ), "TODO 2: fix the row-count expression in summary_sql."

    metrics = con.sql(summary_sql).df()
    delete_prefix(client, bucket, GOLD_METRICS_KEY)
    con.sql(f"COPY ({summary_sql}) TO '{metrics_uri}' (FORMAT PARQUET)")
    reread_rows = con.sql(
        f"select count(*) from read_parquet('{metrics_uri}')"
    ).fetchone()[0]
    if reread_rows != len(metrics):
        raise SystemExit(
            f"Expected {len(metrics)} rows after writing metrics, got {reread_rows}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = write_metric_table(metrics)
    plot = write_metric_plot(metrics)

    print(f"Wrote local metric table: {table}")
    print(f"Wrote plot: {plot}")
    print(f"Wrote Gold metric table: {metrics_uri}")
    print("SQL query:")
    print(summary_sql.strip())
    print(metrics)


if __name__ == "__main__":
    main()

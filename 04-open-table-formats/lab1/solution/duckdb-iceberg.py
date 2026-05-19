from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd
from pandas.api import types as pandas_types

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/preparation-for-ai/04-open-table-formats")

OUTPUT_DIR = Path("/root/preparation-for-ai/04-open-table-formats/lab1/solution")

from iceberg_lab_common import (  # noqa: E402
    full_table_name,
    iceberg_catalog_alias,
    iceberg_namespace,
    load_environment,
    polaris_catalog_name,
    polaris_oauth_token_uri,
    polaris_rest_uri,
    polaris_scope,
    pyiceberg_catalog,
    require_completed,
    require_env,
    require_sql_identifier,
    s3_endpoint_for_duckdb,
    sql_string,
)

# TODO:
# Use the same table name you created with PyIceberg in Step 1.
TABLE_NAME = "yellow_taxi_trips"


PROFILE_COLUMNS = [
    "pickup_at",
    "dropoff_at",
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "trip_month",
]

PAYMENT_TYPE_LABELS = {
    0: "Unknown",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}

COLUMN_LABELS = {
    "pickup_at": "Pickup day",
    "dropoff_at": "Dropoff day",
    "passenger_count": "Passenger count",
    "trip_distance": "Trip distance",
    "pickup_location_id": "Pickup location ID",
    "dropoff_location_id": "Dropoff location ID",
    "payment_type": "Payment type",
    "fare_amount": "Fare amount",
    "tip_amount": "Tip amount",
    "total_amount": "Total amount",
    "trip_month": "Trip month",
}


def payment_type_label(value: object) -> str:
    try:
        payment_id = int(value)
    except (TypeError, ValueError):
        return str(value)
    name = PAYMENT_TYPE_LABELS.get(payment_id, "Other")
    return f"{payment_id}: {name}"


def day_label(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).split()[0]


def write_column_range_plot(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "duckdb1_column_ranges.png"

    fig, axes = plt.subplots(4, 3, figsize=(16, 13))
    axes = axes.flatten()

    for index, column in enumerate(PROFILE_COLUMNS):
        axis = axes[index]
        series = frame[column].dropna()

        if pandas_types.is_datetime64_any_dtype(series):
            counts = series.dt.date.value_counts().sort_index()
            labels = [value.isoformat() for value in counts.index]
            axis.bar(labels, counts.values, color="#2563eb")
            axis.tick_params(axis="x", rotation=55, labelsize=7)
            axis.set_xlabel("day")
        elif pandas_types.is_numeric_dtype(series) and series.nunique() > 15:
            axis.hist(series, bins=30, color="#059669")
            axis.set_xlabel(COLUMN_LABELS[column])
        else:
            counts = series.value_counts().sort_index()
            if column == "payment_type":
                labels = [payment_type_label(value) for value in counts.index]
            else:
                labels = [str(value) for value in counts.index]
            axis.bar(labels, counts.values, color="#7c3aed")
            axis.set_xlabel(COLUMN_LABELS[column])
            axis.tick_params(axis="x", rotation=30, labelsize=7)

        axis.set_title(COLUMN_LABELS[column])
        axis.set_ylabel("Rows")
        axis.grid(axis="y", alpha=0.2)

    axes[-1].axis("off")
    fig.suptitle("DuckDB Column Ranges In The Iceberg Table", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_daily_plot(report: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "duckdb2_daily_overview.png"
    daily = report["daily_rows"]
    labels = [day_label(row["trip_day"]) for row in daily]

    metrics = [
        ("rows", "Rows"),
        ("average_distance", "Average Distance"),
        ("average_total", "Average Total Amount"),
        ("average_tip", "Average Tip Amount"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.flatten()
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]
    for axis, (field, title), color in zip(axes, metrics, colors, strict=True):
        axis.bar(labels, [row[field] for row in daily], color=color)
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=55, labelsize=7)

    fig.suptitle(f"DuckDB Daily Metrics: {report['rows']} Rows", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_payment_plot(report: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "duckdb3_payment_type_overview.png"
    payments = report["payment_type_rows"]
    labels = [payment_type_label(row["payment_type"]) for row in payments]

    metrics = [
        ("rows", "Rows"),
        ("average_distance", "Average Distance"),
        ("average_total", "Average Total Amount"),
        ("average_tip", "Average Tip Amount"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = ["#7c3aed", "#059669", "#dc2626", "#2563eb"]
    for axis, (field, title), color in zip(axes, metrics, colors, strict=True):
        axis.bar(labels, [row[field] for row in payments], color=color)
        axis.set_title(title)
        axis.set_xlabel("Payment type")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle(f"DuckDB Query Result: {report['full_table_name']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def configure_duckdb(connection: duckdb.DuckDBPyConnection) -> None:
    endpoint, use_ssl = s3_endpoint_for_duckdb()
    token_uri = polaris_oauth_token_uri(internal=False)

    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    connection.execute("INSTALL iceberg")
    connection.execute("LOAD iceberg")
    connection.execute(
        f"""
        CREATE OR REPLACE SECRET participant_s3 (
            TYPE s3,
            KEY_ID {sql_string(require_env("S3_ACCESS_KEY_ID"))},
            SECRET {sql_string(require_env("S3_SECRET_ACCESS_KEY"))},
            ENDPOINT {sql_string(endpoint)},
            USE_SSL {'true' if use_ssl else 'false'},
            URL_STYLE 'path',
            REGION 'us-east-1'
        )
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE SECRET polaris_oauth (
            TYPE iceberg,
            CLIENT_ID {sql_string(require_env("POLARIS_CLIENT_ID"))},
            CLIENT_SECRET {sql_string(require_env("POLARIS_CLIENT_SECRET"))},
            OAUTH2_SERVER_URI {sql_string(token_uri)},
            OAUTH2_SCOPE {sql_string(polaris_scope())}
        )
        """
    )


def attach_catalog(connection: duckdb.DuckDBPyConnection) -> None:
    alias = iceberg_catalog_alias()
    connection.execute(
        f"""
        ATTACH {sql_string(polaris_catalog_name())} AS {alias} (
            TYPE iceberg,
            SECRET polaris_oauth,
            ENDPOINT {sql_string(polaris_rest_uri(internal=False))},
            ACCESS_DELEGATION_MODE 'none',
            SUPPORT_NESTED_NAMESPACES false
        )
        """
    )


def current_snapshot_id(table_name: str) -> str:
    catalog = pyiceberg_catalog()
    table = catalog.load_table(f"{iceberg_namespace()}.{table_name}")
    table.refresh()
    snapshot = table.current_snapshot()
    if snapshot is None:
        return "none"
    return str(snapshot.snapshot_id)


def main() -> None:
    load_environment()
    require_completed(TABLE_NAME, "TODO: fill in TABLE_NAME.")
    require_sql_identifier(TABLE_NAME, "TABLE_NAME")

    namespace = iceberg_namespace()
    catalog_alias = iceberg_catalog_alias()
    table = full_table_name(TABLE_NAME)

    connection = duckdb.connect(database=":memory:")
    configure_duckdb(connection)
    attach_catalog(connection)

    rows = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    columns = len(connection.execute(f"SELECT * FROM {table} LIMIT 0").description)
    snapshot_id = current_snapshot_id(TABLE_NAME)
    payment_type_rows = connection.execute(
        f"""
        SELECT
            payment_type,
            count(*) AS rows,
            avg(trip_distance) AS average_distance,
            avg(total_amount) AS average_total,
            avg(tip_amount) AS average_tip
        FROM {table}
        GROUP BY payment_type
        ORDER BY rows DESC
        """
    ).fetchdf()
    daily_rows = connection.execute(
        f"""
        SELECT
            CAST(pickup_at AS DATE) AS trip_day,
            count(*) AS rows,
            avg(trip_distance) AS average_distance,
            avg(total_amount) AS average_total,
            avg(tip_amount) AS average_tip
        FROM {table}
        GROUP BY trip_day
        ORDER BY trip_day
        """
    ).fetchdf()
    profile_frame = connection.execute(
        f"SELECT {', '.join(PROFILE_COLUMNS)} FROM {table}"
    ).fetchdf()

    report = {
        "full_table_name": table,
        "polaris_catalog": polaris_catalog_name(),
        "catalog_alias": catalog_alias,
        "namespace": namespace,
        "table": TABLE_NAME,
        "rows": int(rows),
        "columns": int(columns),
        "snapshot_id": snapshot_id,
        "payment_type_rows": payment_type_rows.to_dict("records"),
        "daily_rows": daily_rows.to_dict("records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    column_plot = write_column_range_plot(profile_frame)
    daily_plot = write_daily_plot(report)
    payment_plot = write_payment_plot(report)

    print(f"Queried Iceberg table: {table}")
    print(f"Rows: {rows}")
    print(f"Current snapshot ID: {snapshot_id}")
    print(f"Wrote plot: {column_plot}")
    print(f"Wrote plot: {daily_plot}")
    print(f"Wrote plot: {payment_plot}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import duckdb
import polars as pl

PAYMENT_TYPE_LABELS = {
    0: "Unknown",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--input-month", required=True)
    parser.add_argument("--minimum-rows", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://s3.mechatronics.lan"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def download_input(client, bucket: str, month: str, path: Path) -> str:
    key = f"bronze/chapter04/nyc-taxi/yellow_taxi_{month}_sample.parquet"
    print(f"Downloading s3://{bucket}/{key}")
    client.download_file(bucket, key, str(path))
    return key


def upload_file(client, bucket: str, prefix: str, path: Path) -> None:
    key = f"{prefix.rstrip('/')}/{path.name}"
    print(f"Uploading s3://{bucket}/{key}")
    client.upload_file(str(path), bucket, key)


def clean_with_polars(input_path: Path, output_path: Path, month: str) -> int:
    print("Cleaning taxi rows with Polars")
    silver = (
        pl.scan_parquet(input_path)
        .filter(
            (pl.col("trip_distance") > 0)
            & (pl.col("trip_distance") < 100)
            & (pl.col("total_amount") > 0)
            & (pl.col("total_amount") < 500)
            & (
                pl.col("pickup_at")
                .dt.strftime("%Y-%m")
                .eq(pl.lit(month))
            )
        )
        .with_columns(
            [
                pl.col("pickup_at").dt.hour().alias("pickup_hour"),
                (
                    (
                        pl.col("dropoff_at")
                        - pl.col("pickup_at")
                    ).dt.total_seconds()
                    / 60.0
                )
                .round(2)
                .alias("trip_minutes"),
                pl.when(pl.col("fare_amount") > 0)
                .then(pl.col("tip_amount") / pl.col("fare_amount"))
                .otherwise(0.0)
                .round(4)
                .alias("tip_rate"),
            ]
        )
        .select(
            [
                "pickup_at",
                "dropoff_at",
                "pickup_hour",
                "passenger_count",
                "trip_distance",
                "trip_minutes",
                "fare_amount",
                "tip_amount",
                "tip_rate",
                "total_amount",
                "payment_type",
            ]
        )
        .collect()
    )
    silver.write_parquet(output_path)
    return silver.height


def build_metrics_with_duckdb(silver_path: Path, output_dir: Path) -> None:
    print("Building metric tables with DuckDB SQL")
    connection = duckdb.connect()
    connection.execute(
        """
        create or replace table taxi as
        select
            *,
            case payment_type
                when 1 then 'Credit card'
                when 2 then 'Cash'
                when 3 then 'No charge'
                when 4 then 'Dispute'
                when 6 then 'Voided trip'
                else 'Unknown'
            end as payment_type_label
        from read_parquet(?)
        """,
        [str(silver_path)],
    )
    connection.execute(
        """
        copy (
            select
                pickup_hour,
                count(*) as trips,
                round(avg(trip_distance), 2) as average_distance,
                round(avg(trip_minutes), 2) as average_minutes,
                round(avg(total_amount), 2) as average_total,
                round(avg(tip_rate), 4) as average_tip_rate
            from taxi
            group by pickup_hour
            order by pickup_hour
        ) to ? (format parquet)
        """,
        [str(output_dir / "gold_hourly_metrics.parquet")],
    )
    connection.execute(
        """
        copy (
            select
                payment_type,
                payment_type_label,
                count(*) as trips,
                round(avg(trip_distance), 2) as average_distance,
                round(avg(total_amount), 2) as average_total,
                round(avg(tip_rate), 4) as average_tip_rate
            from taxi
            group by payment_type, payment_type_label
            order by trips desc
        ) to ? (format parquet)
        """,
        [str(output_dir / "gold_payment_metrics.parquet")],
    )
    connection.execute(
        """
        copy (
            select
                case
                    when trip_distance < 1 then '0-1 miles'
                    when trip_distance < 3 then '1-3 miles'
                    when trip_distance < 8 then '3-8 miles'
                    when trip_distance < 20 then '8-20 miles'
                    else '20+ miles'
                end as distance_bucket,
                count(*) as trips,
                round(avg(total_amount), 2) as average_total,
                round(avg(tip_rate), 4) as average_tip_rate
            from taxi
            group by distance_bucket
            order by min(trip_distance)
        ) to ? (format parquet)
        """,
        [str(output_dir / "gold_distance_metrics.parquet")],
    )
    connection.close()


def write_report(path: Path, args: argparse.Namespace, input_key: str, rows: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Chapter 6 Taxi Python Report",
                "",
                f"Input: `s3://{args.shared_bucket}/{input_key}`",
                f"Output prefix: `s3://{args.output_bucket}/{args.output_prefix}`",
                f"Input month: `{args.input_month}`",
                f"Cleaned rows: `{rows}`",
                f"Required minimum rows: `{args.minimum_rows}`",
                "",
                "Tools used:",
                "",
                "- Polars cleaned and selected the row-level taxi dataset.",
                "- DuckDB SQL built the Gold metric tables.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    client = s3_client()
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        input_path = work_dir / "input.parquet"
        silver_path = work_dir / "silver_taxi_trips.parquet"
        input_key = download_input(client, args.shared_bucket, args.input_month, input_path)
        rows = clean_with_polars(input_path, silver_path, args.input_month)

        print(f"Cleaned rows: {rows}")
        print(f"Required minimum rows: {args.minimum_rows}")
        if rows < args.minimum_rows:
            raise RuntimeError(
                f"Validation failed: cleaned row count {rows} is below "
                f"minimum_rows={args.minimum_rows}."
            )

        build_metrics_with_duckdb(silver_path, work_dir)
        report_path = work_dir / "taxi1_run_report.md"
        write_report(report_path, args, input_key, rows)

        for path in [
            report_path,
            silver_path,
            work_dir / "gold_hourly_metrics.parquet",
            work_dir / "gold_payment_metrics.parquet",
            work_dir / "gold_distance_metrics.parquet",
        ]:
            upload_file(client, args.output_bucket, args.output_prefix, path)

    print("Taxi workflow completed successfully.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession, functions as F, types as T

SECRET_REDACTION_REGEX = "(?i)secret|password|token|credential|access[.]?key"


def require_argument(index: int, name: str) -> str:
    try:
        return sys.argv[index]
    except IndexError as exc:
        raise SystemExit(f"Missing argument: {name}") from exc


def require_secret_environment() -> tuple[str, str, str, str, str, str]:
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    polaris_client_id = os.getenv("POLARIS_CLIENT_ID")
    polaris_client_secret = os.getenv("POLARIS_CLIENT_SECRET")
    scope = os.getenv("POLARIS_SCOPE", "PRINCIPAL_ROLE:ALL")
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://s3.mechatronics.lan")

    missing = [
        name
        for name, value in [
            ("AWS_ACCESS_KEY_ID", access_key),
            ("AWS_SECRET_ACCESS_KEY", secret_key),
            ("POLARIS_CLIENT_ID", polaris_client_id),
            ("POLARIS_CLIENT_SECRET", polaris_client_secret),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing secret-backed environment variables: {missing}")

    return access_key, secret_key, polaris_client_id, polaris_client_secret, scope, endpoint


def build_spark_session(catalog_name: str) -> SparkSession:
    (
        access_key,
        secret_key,
        polaris_client_id,
        polaris_client_secret,
        scope,
        endpoint,
    ) = require_secret_environment()

    return (
        SparkSession.builder.appName("chapter04-iceberg-table-job")
        .config("spark.redaction.regex", SECRET_REDACTION_REGEX)
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            f"spark.sql.catalog.{catalog_name}.credential",
            f"{polaris_client_id}:{polaris_client_secret}",
        )
        .config(f"spark.sql.catalog.{catalog_name}.scope", scope)
        .config(f"spark.sql.catalog.{catalog_name}.s3.access-key-id", access_key)
        .config(f"spark.sql.catalog.{catalog_name}.s3.secret-access-key", secret_key)
        .config(f"spark.sql.catalog.{catalog_name}.s3.endpoint", endpoint)
        .config(f"spark.sql.catalog.{catalog_name}.s3.path-style-access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.connection.ssl.enabled",
            "true" if endpoint.startswith("https://") else "false",
        )
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .getOrCreate()
    )


def configure_catalog_runtime(spark: SparkSession, catalog_name: str) -> None:
    (
        access_key,
        secret_key,
        polaris_client_id,
        polaris_client_secret,
        scope,
        endpoint,
    ) = require_secret_environment()

    spark.conf.set(
        f"spark.sql.catalog.{catalog_name}.credential",
        f"{polaris_client_id}:{polaris_client_secret}",
    )
    spark.conf.set(f"spark.sql.catalog.{catalog_name}.scope", scope)
    spark.conf.set(f"spark.sql.catalog.{catalog_name}.s3.access-key-id", access_key)
    spark.conf.set(f"spark.sql.catalog.{catalog_name}.s3.secret-access-key", secret_key)
    spark.conf.set(f"spark.sql.catalog.{catalog_name}.s3.endpoint", endpoint)
    spark.conf.set(f"spark.sql.catalog.{catalog_name}.s3.path-style-access", "true")

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.endpoint", endpoint)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set(
        "fs.s3a.connection.ssl.enabled",
        "true" if endpoint.startswith("https://") else "false",
    )
    hadoop_conf.set("fs.s3a.access.key", access_key)
    hadoop_conf.set("fs.s3a.secret.key", secret_key)


def clean_taxi_frame(spark: SparkSession, source_uri: str, month: str):
    return (
        spark.read.parquet(source_uri)
        .select(
            F.col("pickup_at").cast(T.TimestampType()).alias("pickup_at"),
            F.col("dropoff_at").cast(T.TimestampType()).alias("dropoff_at"),
            F.col("passenger_count").cast("int").alias("passenger_count"),
            F.col("trip_distance").cast("double").alias("trip_distance"),
            F.col("pickup_location_id").cast("int").alias("pickup_location_id"),
            F.col("dropoff_location_id").cast("int").alias("dropoff_location_id"),
            F.col("payment_type").cast("int").alias("payment_type"),
            F.col("fare_amount").cast("double").alias("fare_amount"),
            F.col("tip_amount").cast("double").alias("tip_amount"),
            F.col("total_amount").cast("double").alias("total_amount"),
            F.col("trip_month").cast("string").alias("trip_month"),
        )
        .where(F.date_format(F.col("pickup_at"), "yyyy-MM") == month)
    )


def main() -> None:
    catalog_name = require_argument(1, "Iceberg catalog name")
    table_namespace = require_argument(2, "Iceberg table namespace")
    table_name = require_argument(3, "Iceberg table name")
    february_input_uri = require_argument(4, "February taxi input URI")

    full_table_name = f"{catalog_name}.{table_namespace}.{table_name}"

    spark = build_spark_session(catalog_name)
    configure_catalog_runtime(spark, catalog_name)

    before = spark.table(full_table_name)
    rows_before = before.count()
    columns_before = len(before.columns)

    february = clean_taxi_frame(spark, february_input_uri, "2024-02")
    appended_rows = february.count()
    february.writeTo(full_table_name).append()

    after = spark.table(full_table_name)
    rows_after = after.count()
    columns_after = len(after.columns)

    print(f"Processed Iceberg table: {full_table_name}")
    print(f"Rows before: {rows_before}")
    print(f"Columns before: {columns_before}")
    print(f"Appended February rows: {appended_rows}")
    print(f"Rows after: {rows_after}")
    print(f"Columns after: {columns_after}")
    print("Rerun the DuckDB script to inspect the new current snapshot ID.")
    spark.stop()


if __name__ == "__main__":
    main()

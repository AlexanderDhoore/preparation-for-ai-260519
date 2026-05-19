from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession, functions as F


def require_argument(index: int, name: str) -> str:
    try:
        return sys.argv[index]
    except IndexError as exc:
        raise SystemExit(f"Missing argument: {name}") from exc


def configure_s3(spark: SparkSession) -> None:
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://s3.mechatronics.lan")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise SystemExit(
            "Missing S3 credentials. The SparkApplication should expose "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from a Kubernetes secret."
        )

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


def main() -> None:
    source_uri = require_argument(1, "source Parquet URI")
    output_uri = require_argument(2, "output Parquet URI")

    spark = (
        SparkSession.builder.appName("chapter03-spark-covertype-summary")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    configure_s3(spark)

    silver = spark.read.parquet(source_uri)
    summary = (
        silver.groupBy("cover_type")
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("wilderness_area").alias("wilderness_areas"),
            F.countDistinct("soil_type").alias("soil_types"),
            F.round(F.avg("elevation"), 2).alias("avg_elevation"),
            F.round(F.min("elevation"), 2).alias("min_elevation"),
            F.round(F.max("elevation"), 2).alias("max_elevation"),
        )
        .orderBy("cover_type")
    )

    print("Spark metric table:")
    summary.show(50, truncate=False)
    summary.coalesce(1).write.mode("overwrite").parquet(output_uri)

    print(f"Read Silver Parquet: {source_uri}")
    print(f"Wrote Spark metric table: {output_uri}")
    spark.stop()


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys

sys.path.insert(0, "/root/preparation-for-ai/04-open-table-formats")

from iceberg_lab_common import (  # noqa: E402
    build_spark_application,
    full_table_name,
    iceberg_catalog_alias,
    iceberg_namespace,
    print_manifest_instructions,
    require_completed,
    require_sql_identifier,
    shared_taxi_s3a_uri,
)

# TODO:
# Use the same table name you created with PyIceberg in Step 1.
TABLE_NAME = "TODO_FILL_TABLE_NAME"


def main() -> None:
    require_completed(TABLE_NAME, "TODO: fill in TABLE_NAME.")
    require_sql_identifier(TABLE_NAME, "TABLE_NAME")

    catalog_alias = iceberg_catalog_alias()
    namespace = iceberg_namespace()
    february_input_uri = shared_taxi_s3a_uri("2024-02")

    manifest_path, metadata = build_spark_application(
        app_name="chapter04-iceberg-spark",
        config_map_name="chapter04-iceberg-spark-code",
        script_name="spark_chapter04_table_job.py",
        driver_args=[
            catalog_alias,
            namespace,
            TABLE_NAME,
            february_input_uri,
        ],
        report_name="spark_iceberg_table_application",
    )

    print_manifest_instructions(manifest_path, metadata)
    print("")
    print(f"Iceberg table: {full_table_name(TABLE_NAME)}")
    print(f"February input: {february_input_uri}")


if __name__ == "__main__":
    main()

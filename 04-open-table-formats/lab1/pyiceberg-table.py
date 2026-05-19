from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/preparation-for-ai/04-open-table-formats")

OUTPUT_DIR = Path("/root/preparation-for-ai/04-open-table-formats/lab1")

from iceberg_lab_common import (  # noqa: E402
    delete_prefix,
    full_table_name,
    iceberg_namespace,
    iceberg_namespace_location,
    iceberg_table_identifier,
    iceberg_table_location,
    iceberg_table_location_prefix,
    load_environment,
    participant_bucket,
    pyiceberg_catalog,
    require_completed,
    require_sql_identifier,
    s3_client,
    shared_taxi_uri,
    source_taxi_arrow_table,
)

# TODO:
# Choose the Iceberg table name for the taxi trip table.
TABLE_NAME = "TODO_FILL_TABLE_NAME"
SOURCE_MONTH = "2024-01"


def snapshot_report(snapshot) -> dict[str, object] | None:
    if snapshot is None:
        return None
    summary = snapshot.summary or {}
    if hasattr(summary, "model_dump"):
        summary_payload = summary.model_dump(mode="json")
    else:
        summary_payload = dict(summary)
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "parent_snapshot_id": (
            None
            if snapshot.parent_snapshot_id is None
            else str(snapshot.parent_snapshot_id)
        ),
        "operation": summary_payload.get("operation"),
        "summary": summary_payload,
    }


def write_created_table_plot(report: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "pyiceberg1_created_table.png"
    snapshot = report["current_snapshot"] or {}
    column_text = ", ".join(report["column_names"])
    note = "\n".join(
        [
            f"current snapshot: {snapshot.get('snapshot_id')}",
            "columns:",
            *textwrap.wrap(column_text, width=95),
        ]
    )

    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(2, 1, height_ratios=[4, 1.25])
    axis = fig.add_subplot(grid[0, 0])
    note_axis = fig.add_subplot(grid[1, 0])

    rows = [report["rows_written"], report["rows_read_back"]]
    axis.bar(["written", "read back"], rows, color=["#2563eb", "#059669"])
    axis.set_title("Rows In The Iceberg Table")
    axis.set_ylabel("Rows")
    axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"PyIceberg Created {report['table']} From {report['source_month']}",
        fontsize=14,
    )
    note_axis.axis("off")
    note_axis.text(
        0.01,
        0.95,
        note,
        transform=note_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    load_environment()
    require_completed(TABLE_NAME, "TODO: fill in TABLE_NAME.")
    require_sql_identifier(TABLE_NAME, "TABLE_NAME")

    namespace = iceberg_namespace()
    identifier = iceberg_table_identifier(TABLE_NAME)
    table_location = iceberg_table_location(TABLE_NAME)
    table_prefix = iceberg_table_location_prefix(TABLE_NAME)
    source_uri = shared_taxi_uri(SOURCE_MONTH)

    catalog = pyiceberg_catalog()
    if not catalog.namespace_exists(namespace):
        catalog.create_namespace(
            namespace,
            properties={"location": iceberg_namespace_location()},
        )

    client = s3_client()
    bucket = participant_bucket()
    if catalog.table_exists(identifier):
        catalog.drop_table(identifier)
    deleted_existing_objects = delete_prefix(client, bucket, table_prefix)

    source = source_taxi_arrow_table(SOURCE_MONTH)
    table = catalog.create_table(
        identifier,
        schema=source.schema,
        location=table_location,
        properties={"format-version": "2"},
    )

    # TODO:
    # Append the Arrow table to the Iceberg table.
    # Hint: use the variable that was loaded from the shared taxi Parquet file.
    table.append(
        TODO_FILL_ARROW_TABLE,
        snapshot_properties={
            "chapter": "04-open-table-formats",
            "source": "nyc-tlc-yellow-taxi",
            "source_month": SOURCE_MONTH,
        },
    )
    table.refresh()
    read_back = table.scan().to_arrow()

    report = {
        "full_table_name": full_table_name(TABLE_NAME),
        "pyiceberg_identifier": identifier,
        "namespace": namespace,
        "table": TABLE_NAME,
        "source_month": SOURCE_MONTH,
        "source_uri": source_uri,
        "table_location": table.location(),
        "metadata_location": table.metadata_location,
        "rows_written": source.num_rows,
        "rows_read_back": read_back.num_rows,
        "columns": len(source.column_names),
        "column_names": source.column_names,
        "deleted_existing_objects": deleted_existing_objects,
        "current_snapshot": snapshot_report(table.current_snapshot()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created_table_plot = write_created_table_plot(report)

    print(f"Created Iceberg table: {report['full_table_name']}")
    print(f"Rows written: {source.num_rows}")
    print(f"Rows read back: {read_back.num_rows}")
    print(f"Wrote plot: {created_table_plot}")


if __name__ == "__main__":
    main()

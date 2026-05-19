from __future__ import annotations

import io
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CHAPTER_DIR = Path("/root/preparation-for-ai/01-ai-ready-data")


@dataclass(frozen=True)
class PublicFile:
    dataset: str
    url: str
    output_path: Path
    description: str
    license_note: str


FILES = [
    PublicFile(
        dataset="mstz/covertype",
        url=(
            "https://huggingface.co/datasets/mstz/covertype/"
            "resolve/refs%2Fconvert%2Fparquet/covertype/train/0000.parquet"
        ),
        output_path=CHAPTER_DIR / "covertype.csv",
        description="Forest cover type classification data.",
        license_note="Check the Hugging Face dataset card before reuse outside this workshop.",
    ),
    PublicFile(
        dataset="nominal-io/nasa-turbofan-degradation",
        url=(
            "https://huggingface.co/datasets/nominal-io/"
            "nasa-turbofan-degradation/resolve/main/NASA_turbofan_train_FD001.csv"
        ),
        output_path=CHAPTER_DIR / "turbofan.csv",
        description="NASA C-MAPSS FD001 train split for remaining-useful-life regression.",
        license_note="MIT, according to the Hugging Face dataset card",
    ),
]


def download_bytes(file: PublicFile) -> bytes:
    print(f"Downloading {file.url}")
    with urllib.request.urlopen(file.url, timeout=60) as response:
        return response.read()


def collapse_indicator_columns(
    frame: pd.DataFrame, prefix: str, output_column: str
) -> pd.DataFrame:
    indicator_columns = [
        column for column in frame.columns if column.startswith(prefix)
    ]
    if not indicator_columns:
        raise ValueError(f"No columns found with prefix {prefix!r}")

    active_counts = frame[indicator_columns].sum(axis=1)
    if not active_counts.eq(1).all():
        raise ValueError(
            f"Expected exactly one active column for {output_column} in every row."
        )

    collapsed = frame[indicator_columns].idxmax(axis=1).str.removeprefix(prefix)
    insert_at = frame.columns.get_loc(indicator_columns[0])
    frame = frame.drop(columns=indicator_columns)
    frame.insert(insert_at, output_column, collapsed.astype("int16"))
    return frame


def prepare_covertype_dataset() -> None:
    public_file = FILES[0]
    csv_path = public_file.output_path

    if csv_path.is_file():
        print(f"Using cached {csv_path}")
        return

    print(f"Preparing {csv_path}")
    frame = pd.read_parquet(io.BytesIO(download_bytes(public_file)))
    frame = collapse_indicator_columns(frame, "wilderness_area_id_", "wilderness_area")
    frame = collapse_indicator_columns(frame, "soil_type_id_", "soil_type")
    frame.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({csv_path.stat().st_size} bytes)")


def normalize_name(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("◦", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def prepare_turbofan_dataset() -> None:
    public_file = FILES[1]
    csv_path = public_file.output_path

    if csv_path.is_file():
        print(f"Using cached {csv_path}")
        return

    print(f"Preparing {csv_path}")
    frame = pd.read_csv(io.BytesIO(download_bytes(public_file)))
    frame.columns = [normalize_name(column) for column in frame.columns]
    if "unnamed_0" in frame.columns:
        frame = frame.drop(columns=["unnamed_0"])
    frame.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({csv_path.stat().st_size} bytes)")


def main() -> None:
    prepare_covertype_dataset()
    prepare_turbofan_dataset()


if __name__ == "__main__":
    main()

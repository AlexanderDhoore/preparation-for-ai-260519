from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path("/root/preparation-for-ai")
DATASET_DIR = REPO_ROOT / "01-ai-ready-data"
OUTPUT_DIR = REPO_ROOT / "01-ai-ready-data" / "lab1"
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
# Look at `01-ai-ready-data/covertype.csv` and find the target column.
# Hint: it is the last column in the CSV file.
TARGET_COLUMN = "TODO_FILL_TARGET_COLUMN"


def require_completed(value: object, message: str) -> None:
    if isinstance(value, str) and value.startswith("TODO"):
        raise NotImplementedError(message)


def load_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "covertype.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run `python3 01-ai-ready-data/download_datasets.py` "
            "before this lab."
        )
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame) -> None:
    require_completed(TARGET_COLUMN, "TODO 1: set TARGET_COLUMN.")
    if TARGET_COLUMN not in frame.columns:
        raise SystemExit(
            "The selected target column does not exist.\n"
            f"Target: {TARGET_COLUMN!r}\n"
            f"Available columns: {list(frame.columns)}"
        )


def target_label(value: object) -> str:
    if value in TARGET_LABELS:
        return f"{value} = {TARGET_LABELS[value]}"
    return str(value)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != TARGET_COLUMN]


def continuous_feature_columns(frame: pd.DataFrame) -> list[str]:
    categorical_columns = {"wilderness_area", "soil_type"}
    return [
        column for column in feature_columns(frame) if column not in categorical_columns
    ]


def categorical_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in ["wilderness_area", "soil_type"] if column in frame.columns
    ]


def write_target_distribution(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore1_target_distribution.png"
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    labels = [target_label(value) for value in counts.index]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, counts.values, color="#2563eb")
    ax.set_title("Forest cover type class balance")
    ax.set_xlabel("Cover type")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_continuous_overview(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore2_continuous_overview.png"
    columns = continuous_feature_columns(frame)
    column_count = 3
    row_count = (len(columns) + column_count - 1) // column_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(13, row_count * 3))
    axes = list(axes.ravel())
    for axis, column in zip(axes, columns, strict=False):
        axis.hist(frame[column].dropna(), bins=40, color="#2563eb", alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("Rows")
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Continuous input-column distributions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2, w_pad=1.4)
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_continuous_target_grid(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore3_continuous_vs_target.png"
    columns = continuous_feature_columns(frame)
    labels = sorted(frame[TARGET_COLUMN].dropna().unique())

    column_count = 2
    row_count = (len(columns) + column_count - 1) // column_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(13, row_count * 3.5))
    axes = list(axes.ravel())
    for axis, column in zip(axes, columns, strict=False):
        data = [
            frame.loc[frame[TARGET_COLUMN] == label, column].dropna()
            for label in labels
        ]
        axis.boxplot(
            data,
            tick_labels=[str(label) for label in labels],
            patch_artist=True,
        )
        axis.set_title(column)
        axis.set_xlabel(TARGET_COLUMN)
        axis.set_ylabel("Value")
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Continuous columns compared with cover type", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2, w_pad=1.4)
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_categorical_overview(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore4_categorical_overview.png"
    columns = categorical_feature_columns(frame)
    fig, axes = plt.subplots(1, len(columns), figsize=(16, 5.2))
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns, strict=False):
        counts = frame[column].value_counts().sort_index()
        axis.bar(counts.index.astype(str), counts.values, color="#059669")
        axis.set_title(column)
        axis.set_xlabel("Category")
        axis.set_ylabel("Rows")
        if len(counts) > 12:
            axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.suptitle("Categorical input-column distributions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    validate_columns(frame)
    plots = {
        "target_distribution": str(write_target_distribution(frame)),
        "continuous_overview": str(write_continuous_overview(frame)),
        "continuous_vs_target": str(write_continuous_target_grid(frame)),
        "categorical_overview": str(write_categorical_overview(frame)),
    }

    for plot in plots.values():
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()

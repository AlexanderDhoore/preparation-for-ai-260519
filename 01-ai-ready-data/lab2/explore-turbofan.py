from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path("/root/preparation-for-ai")
DATASET_PATH = REPO_ROOT / "01-ai-ready-data" / "turbofan.csv"
OUTPUT_DIR = REPO_ROOT / "01-ai-ready-data" / "lab2"


def example_engines(frame: pd.DataFrame) -> list[int]:
    lengths = frame.groupby("engine")["cycle"].max().sort_values()
    positions = [0, len(lengths) // 2, len(lengths) - 1]
    return [int(lengths.index[position]) for position in positions]


def changing_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"engine", "cycle"} and frame[column].nunique() > 1
    ]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise SystemExit(
            f"Missing {DATASET_PATH}. Run `python3 01-ai-ready-data/download_datasets.py` "
            "before this lab."
        )
    return pd.read_csv(DATASET_PATH)


def write_engine_lengths_plot(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore1_engine_cycle_counts.png"
    cycle_counts = frame.groupby("engine")["cycle"].max().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(cycle_counts)), cycle_counts.values, color="#2563eb")
    ax.set_title("Recorded life length per engine")
    ax.set_xlabel("Engines sorted by final cycle")
    ax.set_ylabel("Final recorded cycle")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_metric_trajectory_plot(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "explore2_metric_trajectories.png"
    engines = example_engines(frame)
    columns = changing_metric_columns(frame)
    column_count = 3
    row_count = (len(columns) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, max(4.5, row_count * 2.9)),
        squeeze=False,
    )
    flattened_axes = list(axes.ravel())

    for axis, column in zip(flattened_axes, columns, strict=False):
        for engine in engines:
            trajectory = frame[frame["engine"] == engine].sort_values("cycle")
            axis.plot(
                trajectory["cycle"],
                trajectory[column],
                linewidth=1.35,
                alpha=0.8,
                label=f"engine {engine}",
            )
        axis.set_title(column)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.2)

    for axis in flattened_axes[len(columns) :]:
        axis.axis("off")

    flattened_axes[0].legend(loc="best")
    fig.suptitle("Metric trajectories for three example engines", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.1)
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    plots = {
        "engine_lengths": write_engine_lengths_plot(frame),
        "metric_trajectories": write_metric_trajectory_plot(frame),
    }
    for plot in plots.values():
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()

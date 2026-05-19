from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

REPO_ROOT = Path("/root/preparation-for-ai")
DATASET_PATH = REPO_ROOT / "01-ai-ready-data" / "turbofan.csv"
OUTPUT_DIR = REPO_ROOT / "01-ai-ready-data" / "lab2"
TARGET = "remaining_useful_life"


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise SystemExit(
            f"Missing {DATASET_PATH}. Run `python3 01-ai-ready-data/download_datasets.py` "
            "before this lab."
        )
    return pd.read_csv(DATASET_PATH)


def add_remaining_useful_life(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # TODO 1:
    # Create the remaining useful life target. This is the number of cycles
    # still left before the engine reaches its final recorded cycle.
    #
    # For each engine, find the final recorded cycle. Then subtract the current
    # cycle from that final cycle.
    #
    # Hint:
    # - `groupby("engine")["cycle"].transform("max")` gives the final cycle
    #   for the engine on every row.
    # - Store final cycle minus current cycle in frame[TARGET].
    raise NotImplementedError("TODO 1: create the remaining useful life target.")

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    # This is intentionally wrong for this script. Keeping `engine` as a feature
    # helps the leaky split look much better than it deserves. You will fix
    # this in the honest model script.
    blocked_columns = {TARGET}
    return [
        column
        for column in frame.columns
        if column not in blocked_columns and frame[column].nunique() > 1
    ]


def random_row_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temporary = train_test_split(
        frame,
        test_size=0.4,
        random_state=20260519,
        shuffle=True,
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=20260519,
        shuffle=True,
    )
    train = train.copy()
    validation = validation.copy()
    test = test.copy()
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"
    return train, validation, test


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, object]:
    model = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.04,
        max_leaf_nodes=63,
        random_state=20260519,
    )
    model.fit(train[features], train[TARGET])

    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    metrics = {}
    predictions = {}
    for split_name, split_frame in split_frames.items():
        split_predictions = model.predict(split_frame[features])
        predictions[split_name] = [float(value) for value in split_predictions]
        metrics[split_name] = {
            "mean_absolute_error_cycles": float(
                mean_absolute_error(split_frame[TARGET], split_predictions)
            ),
        }

    return {
        "features": features,
        "metrics": metrics,
        "test_predictions": predictions["test"],
    }


def example_engines(frame: pd.DataFrame) -> list[int]:
    lengths = frame.groupby("engine")["cycle"].max().sort_values()
    positions = [0, len(lengths) // 2, len(lengths) - 1]
    return [int(lengths.index[position]) for position in positions]


def write_target_plot(frame: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "leaky1_target_trajectory.png"
    engines = example_engines(frame)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for engine in engines:
        trajectory = frame[frame["engine"] == engine].sort_values("cycle")
        ax.plot(
            trajectory["cycle"],
            trajectory[TARGET],
            linewidth=1.8,
            label=f"engine {engine}",
        )
    ax.set_title("Remaining useful life target")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining useful life")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_split_plot(combined: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "leaky2_split_by_engine.png"
    engines = example_engines(combined)
    colors = {"train": "#2563eb", "validation": "#f59e0b", "test": "#d7263d"}
    fig, axes = plt.subplots(len(engines), 1, figsize=(10, 7.2), sharex=True)
    for axis, engine in zip(axes, engines, strict=True):
        trajectory = combined[combined["engine"] == engine].sort_values("cycle")
        for split_name, color in colors.items():
            split_rows = trajectory[trajectory["split"] == split_name]
            axis.scatter(
                split_rows["cycle"],
                split_rows[TARGET],
                s=12,
                alpha=0.75,
                color=color,
                label=split_name,
            )
        axis.set_title(f"engine {engine}")
        axis.set_ylabel("Remaining life")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Cycle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        "Random row split: each example engine is scattered across splits", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.97), h_pad=1.1)
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_prediction_plot(test: pd.DataFrame, result: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "leaky3_predictions.png"
    predictions = result["test_predictions"]
    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    ax.scatter(test[TARGET], predictions, s=10, alpha=0.35, color="#2563eb")
    limit = max(float(test[TARGET].max()), max(predictions))
    ax.plot([0, limit], [0, limit], color="#111827", linestyle="--", linewidth=1)
    ax.set_title("Leaky model: predicted vs actual")
    ax.set_xlabel("Actual remaining useful life")
    ax.set_ylabel("Predicted remaining useful life")
    error = result["metrics"]["test"]["mean_absolute_error_cycles"]
    ax.text(
        0.04,
        0.96,
        f"Average test error: {error:.1f} cycles",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_remaining_useful_life(load_dataset())
    train, validation, test = random_row_split(frame)
    combined = pd.concat([train, validation, test], ignore_index=True)
    result = train_model(train, validation, test, feature_columns(frame))
    plots = {
        "target": write_target_plot(frame),
        "split": write_split_plot(combined),
        "predictions": write_prediction_plot(test, result),
    }
    for plot in plots.values():
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()

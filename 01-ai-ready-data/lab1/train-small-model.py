from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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
# Use the target column you identified in the exploration step.
# Hint: it is the last column in the CSV file.
TARGET = "TODO_FILL_TARGET_COLUMN"

# TODO 2:
# Choose exactly two input columns after inspecting the exploration plots.
# Try to beat the majority-class reference with only these two columns.
FEATURE_COLUMNS = [
    "TODO_CHOOSE_FIRST_INPUT_COLUMN",
    "TODO_CHOOSE_SECOND_INPUT_COLUMN",
]


def require_completed(value: object, message: str) -> None:
    if isinstance(value, str) and value.startswith("TODO"):
        raise NotImplementedError(message)
    if isinstance(value, list) and any(str(item).startswith("TODO") for item in value):
        raise NotImplementedError(message)


def target_name(label: object) -> str:
    return TARGET_LABELS.get(int(label), str(label))


def load_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "covertype.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run `python3 01-ai-ready-data/download_datasets.py` "
            "before this lab."
        )
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame) -> None:
    require_completed(TARGET, "TODO 1: set TARGET to the exact CSV target column.")
    require_completed(FEATURE_COLUMNS, "TODO 2: choose exactly two input columns.")
    if len(FEATURE_COLUMNS) != 2:
        raise AssertionError(
            f"Choose exactly two input columns. Current value: {FEATURE_COLUMNS}"
        )
    missing = {TARGET, *FEATURE_COLUMNS} - set(frame.columns)
    if missing:
        raise SystemExit(
            f"Selected columns are missing: {sorted(missing)}\n"
            f"Available columns: {list(frame.columns)}"
        )
    if TARGET in FEATURE_COLUMNS:
        raise AssertionError("The target column cannot also be an input column.")
    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise AssertionError(f"Feature columns should be numeric: {non_numeric}")


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remaining = train_test_split(
        frame,
        test_size=0.4,
        random_state=20260519,
        stratify=frame[TARGET],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=20260519,
        stratify=remaining[TARGET],
    )
    return train, validation, test


def majority_baseline(test: pd.DataFrame) -> dict[str, object]:
    counts = test[TARGET].value_counts().sort_values(ascending=False)
    label = counts.index[0]
    return {
        "label": int(label),
        "label_name": target_name(label),
        "accuracy": float(counts.iloc[0] / counts.sum()),
    }


def train_model(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=20260519,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET])

    train_predictions = model.predict(train[FEATURE_COLUMNS])
    validation_predictions = model.predict(validation[FEATURE_COLUMNS])
    test_predictions = model.predict(test[FEATURE_COLUMNS])
    labels = sorted(test[TARGET].unique())

    return {
        "model": "RandomForestClassifier",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "train_accuracy": float(accuracy_score(train[TARGET], train_predictions)),
        "validation_accuracy": float(
            accuracy_score(validation[TARGET], validation_predictions)
        ),
        "test_accuracy": float(accuracy_score(test[TARGET], test_predictions)),
        "confusion_matrix": confusion_matrix(
            test[TARGET], test_predictions, labels=labels
        )
        .astype(int)
        .tolist(),
        "label_names": [target_name(label) for label in labels],
    }


def write_accuracy_plot(result: dict[str, object], baseline: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "small1_accuracy.png"
    names = [
        f"baseline\n{baseline['label']} = {baseline['label_name']}",
        "train",
        "validation",
        "test",
    ]
    values = [
        baseline["accuracy"],
        result["train_accuracy"],
        result["validation_accuracy"],
        result["test_accuracy"],
    ]
    colors = ["#6b7280", "#2563eb", "#f59e0b", "#d7263d"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Small model accuracy\n" f"inputs: {FEATURE_COLUMNS[0]} + {FEATURE_COLUMNS[1]}"
    )
    for index, value in enumerate(values):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_confusion_matrix_plot(result: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "small2_confusion_matrix.png"
    matrix = result["confusion_matrix"]
    label_names = result["label_names"]
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Small model test confusion matrix")
    ax.set_xlabel("Predicted cover type")
    ax.set_ylabel("True cover type")
    ax.set_xticks(range(len(label_names)), labels=label_names, rotation=35, ha="right")
    ax.set_yticks(range(len(label_names)), labels=label_names)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > max(map(max, matrix)) * 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    validate_columns(frame)
    train, validation, test = split_dataset(frame)
    result = train_model(train, validation, test)
    baseline = majority_baseline(test)
    accuracy_plot = write_accuracy_plot(result, baseline)
    confusion_matrix_plot = write_confusion_matrix_plot(result)
    print(f"Wrote {accuracy_plot}")
    print(f"Wrote {confusion_matrix_plot}")


if __name__ == "__main__":
    main()

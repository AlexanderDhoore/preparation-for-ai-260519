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
TARGET = "cover_type"
TARGET_LABELS = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}


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


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != TARGET]


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
    features = feature_columns(train)
    model = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=20260519,
        n_jobs=-1,
    )
    model.fit(train[features], train[TARGET])

    train_predictions = model.predict(train[features])
    validation_predictions = model.predict(validation[features])
    test_predictions = model.predict(test[features])
    labels = sorted(test[TARGET].unique())

    return {
        "model": "RandomForestClassifier",
        "target": TARGET,
        "features": features,
        "feature_count": len(features),
        "train_accuracy": float(accuracy_score(train[TARGET], train_predictions)),
        "validation_accuracy": float(
            accuracy_score(validation[TARGET], validation_predictions)
        ),
        "test_accuracy": float(accuracy_score(test[TARGET], test_predictions)),
        "feature_importances": [
            (feature, float(importance))
            for feature, importance in sorted(
                zip(features, model.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "confusion_matrix": confusion_matrix(
            test[TARGET], test_predictions, labels=labels
        )
        .astype(int)
        .tolist(),
        "label_names": [target_name(label) for label in labels],
    }


def write_feature_importance_plot(result: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "big2_feature_importance.png"
    top_features = list(reversed(result["feature_importances"][:12]))
    features = [feature for feature, _ in top_features]
    importances = [importance for _, importance in top_features]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.barh(features, importances, color="#d7263d")
    ax.set_title("Big model feature importances")
    ax.set_xlabel("Random forest importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_confusion_matrix_plot(result: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "big3_confusion_matrix.png"
    matrix = result["confusion_matrix"]
    label_names = result["label_names"]
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Big model test confusion matrix")
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


def write_accuracy_plot(result: dict[str, object], baseline: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "big1_accuracy.png"
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
    ax.set_title(f"Big model accuracy with {result['feature_count']} input columns")
    for index, value in enumerate(values):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    train, validation, test = split_dataset(frame)
    result = train_model(train, validation, test)
    baseline = majority_baseline(test)
    plots = [
        write_accuracy_plot(result, baseline),
        write_feature_importance_plot(result),
        write_confusion_matrix_plot(result),
    ]
    for plot in plots:
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()

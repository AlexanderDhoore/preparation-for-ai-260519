from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/02-object-storage")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from s3_lab_common import load_s3, read_object_bytes, upload_file

OUTPUT_DIR = Path("/root/preparation-for-ai/02-object-storage/lab1")
GOLD_FEATURES_KEY = "gold/chapter02/esc50/audio_features.csv"
ALL_FEATURE_COLUMNS = [
    "rms_energy",
    "peak_amplitude",
    "zero_crossing_rate",
    "mean_absolute_amplitude",
    "spectral_centroid",
    "very_low_frequency_share",
    "low_frequency_share",
    "mid_frequency_share",
    "high_frequency_share",
    "very_high_frequency_share",
]

# TODO 1:
# Choose feature columns after inspecting the Gold feature plots.
# Start with three to six columns, then try a different set and rerun.
FEATURE_COLUMNS = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "low_frequency_share",
    "mid_frequency_share",
    "high_frequency_share",
]


def require_features(columns: list[str], available: set[str]) -> None:
    if not columns or any(column.startswith("TODO") for column in columns):
        raise NotImplementedError("TODO 1: choose feature columns for the model.")
    unknown = sorted(set(columns) - available)
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")


def read_csv_from_s3(client, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_object_bytes(client, bucket, key)))


def fit_model(train: pd.DataFrame, columns: list[str]) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=20260519,
        class_weight="balanced",
    )
    model.fit(train[columns], train["label_text"])
    return model


def write_accuracy_plot(
    majority_reference: float,
    chosen_accuracy: float,
    all_feature_accuracy: float,
) -> Path:
    path = OUTPUT_DIR / "train1_accuracy.png"
    labels = ["Majority\nreference", "Your\nfeatures", "All\nfeatures"]
    values = [majority_reference, chosen_accuracy, all_feature_accuracy]
    figure, axis = plt.subplots(figsize=(7, 4.6))
    bars = axis.bar(labels, values, color=["#9ca3af", "#2563eb", "#4c956c"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Audio Model Accuracy")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_confusion_matrix(
    test: pd.DataFrame,
    chosen_predictions: list[str],
    all_feature_predictions: list[str],
    chosen_accuracy: float,
    all_feature_accuracy: float,
) -> Path:
    path = OUTPUT_DIR / "train2_confusion_matrix.png"
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 14.5))
    labels = sorted(test["label_text"].unique())
    panels = [
        ("Your chosen features", chosen_predictions, chosen_accuracy),
        ("All-feature reference", all_feature_predictions, all_feature_accuracy),
    ]
    for axis, (title, predictions, accuracy) in zip(axes, panels, strict=True):
        display = ConfusionMatrixDisplay.from_predictions(
            test["label_text"],
            predictions,
            labels=labels,
            ax=axis,
            xticks_rotation=35,
            colorbar=False,
        )
        display.ax_.set_title(f"{title} ({accuracy:.1%})")
        display.ax_.set_xlabel("Predicted label")
        display.ax_.set_ylabel("True label")
    figure.suptitle(
        "Audio Classification Confusion Matrices",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.0)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def feature_importance_frame(
    model: RandomForestClassifier,
    columns: list[str],
) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
        .tail(12)
    )


def write_feature_importance(
    chosen_model: RandomForestClassifier,
    chosen_columns: list[str],
    all_feature_model: RandomForestClassifier,
) -> Path:
    path = OUTPUT_DIR / "train3_feature_importance.png"
    panels = [
        (
            "Your chosen features",
            feature_importance_frame(chosen_model, chosen_columns),
            "#2563eb",
        ),
        (
            "All-feature reference",
            feature_importance_frame(all_feature_model, ALL_FEATURE_COLUMNS),
            "#4c956c",
        ),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(9, 10.5))
    for axis, (title, importance, color) in zip(axes, panels, strict=True):
        axis.barh(importance["feature"], importance["importance"], color=color)
        axis.set_title(title)
        axis.set_xlabel("Importance")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Audio Feature Importance", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client, participant_bucket, _shared_bucket = load_s3()
    gold = read_csv_from_s3(client, participant_bucket, GOLD_FEATURES_KEY)
    require_features(FEATURE_COLUMNS, set(gold.columns))

    train = gold[gold["fold"] != 1].copy()
    test = gold[gold["fold"] == 1].copy()
    chosen_model = fit_model(train, FEATURE_COLUMNS)
    all_feature_model = fit_model(train, ALL_FEATURE_COLUMNS)

    chosen_predictions = chosen_model.predict(test[FEATURE_COLUMNS])
    all_feature_predictions = all_feature_model.predict(test[ALL_FEATURE_COLUMNS])
    chosen_accuracy = accuracy_score(test["label_text"], chosen_predictions)
    all_feature_accuracy = accuracy_score(test["label_text"], all_feature_predictions)
    majority_reference = test["label_text"].value_counts(normalize=True).max()

    plots = [
        write_accuracy_plot(
            majority_reference,
            float(chosen_accuracy),
            float(all_feature_accuracy),
        ),
        write_confusion_matrix(
            test,
            list(chosen_predictions),
            list(all_feature_predictions),
            float(chosen_accuracy),
            float(all_feature_accuracy),
        ),
        write_feature_importance(chosen_model, FEATURE_COLUMNS, all_feature_model),
    ]
    for plot in plots:
        upload_file(
            client,
            participant_bucket,
            f"outputs/chapter02/lab1/{plot.name}",
            plot,
            "image/png",
        )

    for plot in plots:
        print(f"Wrote {plot}")
    print(f"Your feature accuracy: {chosen_accuracy:.3f}")
    print(f"All feature accuracy: {all_feature_accuracy:.3f}")


if __name__ == "__main__":
    main()

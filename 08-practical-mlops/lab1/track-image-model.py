from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/08-practical-mlops")

import mlflow
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from mlops_lab_common import (
    build_mobilenet,
    cache_s3_object,
    load_environment,
    load_s3,
    read_manifest,
    require_completed,
    require_env,
    save_confusion_matrix,
    save_prediction_grid,
    save_training_curve,
    write_json,
)

LAB_DIR = Path("/root/preparation-for-ai/08-practical-mlops/lab1")
CACHE_DIR = LAB_DIR / "cache"
BUNDLE_DIR = LAB_DIR / "model_bundle"
MANIFEST_KEY = "datasets/chapter05/food101/manifest.csv"

# TODO 1:
# Pick exactly five Food-101 labels. You can reuse labels from Chapter 5 or
# open 05-multimedia-assets/lab1/explore3_label_examples.png again.
SELECTED_LABELS = [
    "TODO_FILL_LABEL_1",
    "TODO_FILL_LABEL_2",
    "TODO_FILL_LABEL_3",
    "TODO_FILL_LABEL_4",
    "TODO_FILL_LABEL_5",
]

# TODO 2:
# Give your MLflow run a short name. After the first run, change this and one
# training parameter so you can compare two runs in MLflow.
RUN_LABEL = "TODO_FILL_RUN_LABEL"

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Use 0 for the first run. Then try 1 for the second run.
TRAINABLE_BACKBONE_BLOCKS = 0


class ManifestImageDataset(Dataset):
    def __init__(self, frame, client, transform, label_to_target):
        self.frame = frame.reset_index(drop=True)
        self.client = client
        self.transform = transform
        self.label_to_target = label_to_target

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = cache_s3_object(self.client, row.asset_uri, CACHE_DIR / "images")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        target = self.label_to_target[row.label_name]
        return tensor, torch.tensor(target, dtype=torch.long)


def selected_manifest(manifest):
    for label in SELECTED_LABELS:
        require_completed(label, "TODO: choose five Food-101 labels.")
    if len(set(SELECTED_LABELS)) != 5:
        raise ValueError("Choose five different labels.")

    available = set(manifest["label_name"].unique())
    missing = sorted(set(SELECTED_LABELS) - available)
    if missing:
        raise ValueError(f"Unknown Food-101 labels: {missing}")

    subset = manifest.loc[manifest["label_name"].isin(SELECTED_LABELS)].copy()
    subset["label_name"] = subset["label_name"].astype(str)
    return subset


def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # TODO 3:
        # Send the image batch through the neural network.
        # Hint: call the model with the image tensor batch.
        predictions = TODO_CALL_THE_MODEL

        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []
    confidences = []
    with torch.no_grad():
        for images, targets in loader:
            targets = targets.to(device)
            images = images.to(device)
            logits = model(images)
            loss = loss_function(logits, targets)
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            true_labels.extend(targets.cpu().tolist())
            predicted_labels.extend(predictions.tolist())
            confidences.extend(probabilities.max(dim=1).values.tolist())
    return (
        true_labels,
        predicted_labels,
        confidences,
        total_loss / len(loader.dataset),
        accuracy_score(true_labels, predicted_labels),
    )


def save_model_bundle(model, labels: list[str], config: dict) -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), BUNDLE_DIR / "model_state.pt")
    write_json(BUNDLE_DIR / "labels.json", labels)
    write_json(BUNDLE_DIR / "model_config.json", config)


def save_api_sample(client, validation_frame) -> None:
    sample_row = validation_frame.iloc[0]
    source_path = cache_s3_object(client, sample_row.asset_uri, CACHE_DIR / "images")
    with Image.open(source_path) as image:
        image.convert("RGB").save(LAB_DIR / "api_sample_image.jpg", quality=92)


def cache_selected_images(client, manifest) -> None:
    image_count = len(manifest)
    print(f"Downloading and caching {image_count:,} selected images...")
    print("Already cached images are skipped.")
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        cache_s3_object(client, row.asset_uri, CACHE_DIR / "images")
        if index == 1 or index % 250 == 0 or index == image_count:
            print(f"Cached {index:,}/{image_count:,} images", flush=True)
    print("Image cache is ready.")


def main() -> None:
    require_completed(RUN_LABEL, "Fill in RUN_LABEL before running this lab.")

    load_environment()
    participant_id = os.environ.get("PARTICIPANT_ID", "unknown")
    tracking_uri = require_env("MLFLOW_TRACKING_URI")

    client, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, MANIFEST_KEY)
    manifest = selected_manifest(manifest)

    label_names = sorted(manifest["label_name"].unique())
    label_to_target = {label: index for index, label in enumerate(label_names)}
    target_to_label = {index: label for label, index in label_to_target.items()}

    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_mobilenet(len(label_names), TRAINABLE_BACKBONE_BLOCKS)
    model = model.to(device)

    print("Selected food labels:")
    for label in label_names:
        print(f"- {label}")
    print(f"Training examples: {len(train_frame):,}")
    print(f"Validation examples: {len(validation_frame):,}")
    cache_selected_images(client, manifest)
    print(f"Training on {device}")

    train_dataset = ManifestImageDataset(
        train_frame, client, transform, label_to_target
    )
    validation_dataset = ManifestImageDataset(
        validation_frame, client, transform, label_to_target
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()
    history = []

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("preparation-for-ai-chapter-08")

    run_name = f"{participant_id}-{RUN_LABEL}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("participant_id", participant_id)
        mlflow.set_tag("dataset_manifest", f"s3://{shared_bucket}/{MANIFEST_KEY}")
        mlflow.set_tag("script", "08-practical-mlops/lab1/track-image-model.py")
        mlflow.log_param("model", "mobilenet_v3_small")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("trainable_backbone_blocks", TRAINABLE_BACKBONE_BLOCKS)
        mlflow.log_param("selected_labels", ",".join(label_names))
        mlflow.log_param("label_count", len(label_names))
        mlflow.log_param("train_rows", len(train_frame))
        mlflow.log_param("validation_rows", len(validation_frame))

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, optimizer, loss_function, device
            )
            (
                true_labels,
                predicted_labels,
                confidences,
                validation_loss,
                validation_accuracy,
            ) = evaluate(
                model,
                validation_loader,
                loss_function,
                device,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": validation_accuracy,
                }
            )
            mlflow.log_metric("training_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            mlflow.log_metric("training_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("validation_accuracy", validation_accuracy, step=epoch)
            print(
                f"Epoch {epoch}: training loss {train_loss:.3f}, "
                f"training accuracy {train_accuracy:.3f}, "
                f"validation loss {validation_loss:.3f}, "
                f"validation accuracy {validation_accuracy:.3f}"
            )

        final_accuracy = history[-1]["validation_accuracy"]
        mlflow.log_metric("final_validation_accuracy", final_accuracy)
        mlflow.log_metric("final_validation_error", 1.0 - final_accuracy)

        matrix = confusion_matrix(
            true_labels, predicted_labels, labels=range(len(label_names))
        )
        training_curve_path = LAB_DIR / "track1_training_curve.png"
        confusion_matrix_path = LAB_DIR / "track2_confusion_matrix.png"
        predictions_path = LAB_DIR / "track3_predictions.png"

        save_training_curve(history, training_curve_path)
        save_confusion_matrix(matrix, label_names, confusion_matrix_path)

        preview_rows = []
        validation_indexed = validation_frame.reset_index(drop=True)
        for row in (
            validation_indexed.groupby("label_name", sort=True).head(2).itertuples()
        ):
            image_path = cache_s3_object(client, row.asset_uri, CACHE_DIR / "images")
            prediction = predicted_labels[row.Index]
            confidence = confidences[row.Index]
            predicted_label = target_to_label[prediction]
            preview_rows.append(
                (
                    image_path,
                    f"actual: {row.label_name}",
                    f"predicted: {predicted_label}",
                    confidence,
                )
            )
        save_prediction_grid(
            preview_rows, predictions_path, "MLflow prediction examples"
        )

        config = {
            "model_name": "mobilenet_v3_small",
            "label_count": len(label_names),
            "trainable_backbone_blocks": TRAINABLE_BACKBONE_BLOCKS,
            "manifest_key": MANIFEST_KEY,
            "shared_bucket": shared_bucket,
            "mlflow_run_id": run.info.run_id,
            "mlflow_run_name": run_name,
        }
        save_model_bundle(model, label_names, config)
        save_api_sample(client, validation_frame)

        for artifact_path in [
            training_curve_path,
            confusion_matrix_path,
            predictions_path,
            LAB_DIR / "api_sample_image.jpg",
        ]:
            mlflow.log_artifact(str(artifact_path))
        mlflow.log_artifacts(str(BUNDLE_DIR), artifact_path="model_bundle")

    print(f"Wrote {training_curve_path}")
    print(f"Wrote {confusion_matrix_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {LAB_DIR / 'api_sample_image.jpg'}")
    print(f"Wrote {BUNDLE_DIR}")
    print(f"Logged MLflow run: {run_name}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/05-multimedia-assets")

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from multimedia_lab_common import (
    cache_s3_object,
    load_s3,
    read_manifest,
    require_columns,
    require_completed,
    save_class_distribution,
    save_confusion_matrix,
    save_image_preview,
    save_training_curve,
)

LAB_DIR = Path("/root/preparation-for-ai/05-multimedia-assets/lab1")
CACHE_DIR = LAB_DIR / "cache"

MANIFEST_KEY = "datasets/chapter05/food101/manifest.csv"

# TODO 1:
# Pick exactly five labels from explore3_label_examples.png.
# Keep the spelling exactly as it appears in the manifest.
SELECTED_LABELS = [
    "TODO_FILL_LABEL_1",
    "TODO_FILL_LABEL_2",
    "TODO_FILL_LABEL_3",
    "TODO_FILL_LABEL_4",
    "TODO_FILL_LABEL_5",
]

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class FoodImageDataset(Dataset):
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


def build_model(output_classes: int, device: torch.device):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, output_classes)
    return model.to(device), weights.transforms()


def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # TODO 2:
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
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = loss_function(predictions, targets)
            predicted_targets = predictions.argmax(dim=1).cpu()
            total_loss += loss.item() * images.size(0)
            true_labels.extend(targets.cpu().tolist())
            predicted_labels.extend(predicted_targets.tolist())
    accuracy = accuracy_score(true_labels, predicted_labels)
    return true_labels, predicted_labels, total_loss / len(loader.dataset), accuracy


def save_prediction_grid(client, validation_frame, predictions, target_to_label):
    rows = []
    indexed = validation_frame.reset_index(drop=True)
    preview = indexed.groupby("label_name", sort=True).head(8)
    for row in preview.itertuples():
        image_path = cache_s3_object(client, row.asset_uri, CACHE_DIR / "images")
        prediction = predictions[row.Index]
        predicted_label = target_to_label[prediction]
        note_color = (
            (22, 101, 52) if predicted_label == row.label_name else (185, 28, 28)
        )
        rows.append(
            (
                image_path,
                f"actual: {row.label_name}",
                f"predicted: {predicted_label}",
                note_color,
            )
        )
    save_image_preview(rows, LAB_DIR / "food4_predictions.png", "Food predictions")


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
    print("Reading Food-101 manifest from object storage...")
    client, _participant_bucket, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, MANIFEST_KEY)
    require_columns(
        manifest,
        {"asset_uri", "label_name", "split", "sha256", "width", "height"},
        "Food-101 manifest",
    )
    manifest = selected_manifest(manifest)

    label_names = sorted(manifest["label_name"].unique())
    label_to_target = {label: index for index, label in enumerate(label_names)}
    target_to_label = {index: label for label, index in label_to_target.items()}

    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()

    print("Selected food labels:")
    for label in label_names:
        print(f"- {label}")
    print(f"Training examples: {len(train_frame):,}")
    print(f"Validation examples: {len(validation_frame):,}")

    cache_selected_images(client, manifest)
    save_class_distribution(manifest, LAB_DIR / "food1_selected_class_distribution.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_model(len(label_names), device)
    print(f"Training on {device}")
    print("Building PyTorch datasets and dataloaders...")

    train_dataset = FoodImageDataset(train_frame, client, transform, label_to_target)
    validation_dataset = FoodImageDataset(
        validation_frame, client, transform, label_to_target
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        true_labels, predicted_labels, validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device
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
        print(
            f"Epoch {epoch}: training loss {train_loss:.3f}, "
            f"training accuracy {train_accuracy:.3f}, "
            f"validation loss {validation_loss:.3f}, "
            f"validation accuracy {validation_accuracy:.3f}"
        )

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=range(len(label_names))
    )
    save_training_curve(history, LAB_DIR / "food2_training_curve.png")
    save_confusion_matrix(
        matrix,
        label_names,
        LAB_DIR / "food3_confusion_matrix.png",
        "Food-101 validation confusion matrix",
    )
    save_prediction_grid(client, validation_frame, predicted_labels, target_to_label)

    print(f"Final validation accuracy: {validation_accuracy:.3f}")
    print(f"Wrote outputs in {LAB_DIR}")


if __name__ == "__main__":
    main()

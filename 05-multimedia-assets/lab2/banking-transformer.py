from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/05-multimedia-assets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformer_logging

from multimedia_lab_common import (
    cache_s3_object,
    load_s3,
    read_manifest,
    require_columns,
    require_completed,
    save_class_distribution,
    save_confusion_matrix,
    save_training_curve,
    write_markdown_report,
)

LAB_DIR = Path("/root/preparation-for-ai/05-multimedia-assets/lab2")
CACHE_DIR = LAB_DIR / "cache"

MANIFEST_KEY = "datasets/chapter05/banking77/manifest.csv"
MODEL_NAME = "distilbert-base-uncased"
transformer_logging.set_verbosity_error()

# TODO 1:
# Pick exactly five intent labels from explore3_label_menu.md.
# Keep the spelling exactly as it appears in the manifest.
SELECTED_LABELS = [
    "TODO_FILL_LABEL_1",
    "TODO_FILL_LABEL_2",
    "TODO_FILL_LABEL_3",
    "TODO_FILL_LABEL_4",
    "TODO_FILL_LABEL_5",
]

# TODO 2:
# A transformer tokenizer turns text into tokens. Tokens are usually words,
# pieces of words, or punctuation, not individual letters.
# Use explore2_dataset_overview.png to pick a limit that keeps most messages.
MAX_TOKENS = "TODO_FILL_TOKEN_LIMIT"

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.00002


class BankingTextDataset(Dataset):
    def __init__(self, frame, client, tokenizer, max_tokens):
        self.frame = frame.reset_index(drop=True)
        self.client = client
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        text_path = cache_s3_object(self.client, row.asset_uri, CACHE_DIR / "texts")
        text = text_path.read_text(encoding="utf-8")
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        item = {name: value.squeeze(0) for name, value in encoded.items()}
        item["labels"] = torch.tensor(int(row.target), dtype=torch.long)
        return item


def selected_manifest(manifest):
    for label in SELECTED_LABELS:
        require_completed(label, "TODO: choose five Banking77 intent labels.")
    if len(set(SELECTED_LABELS)) != 5:
        raise ValueError("Choose five different intent labels.")

    available = set(manifest["label_name"].unique())
    missing = sorted(set(SELECTED_LABELS) - available)
    if missing:
        raise ValueError(f"Unknown Banking77 labels: {missing}")

    subset = manifest.loc[manifest["label_name"].isin(SELECTED_LABELS)].copy()
    label_names = sorted(subset["label_name"].unique())
    label_to_target = {label: index for index, label in enumerate(label_names)}
    subset["target"] = subset["label_name"].map(label_to_target)
    return subset, label_names


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}

        optimizer.zero_grad()

        output = model(**inputs, labels=labels)

        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (output.logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            output = model(**inputs, labels=labels)
            predictions = output.logits.argmax(dim=1).cpu()
            total_loss += output.loss.item() * labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.tolist())
    accuracy = accuracy_score(true_labels, predicted_labels)
    return true_labels, predicted_labels, total_loss / len(loader.dataset), accuracy


def write_prediction_examples(client, validation_frame, predicted_labels, id_to_label):
    lines = [
        "# Banking77 Prediction Examples",
        "",
        "These examples are from the validation split.",
        "",
    ]
    indexed = validation_frame.reset_index(drop=True)
    sample = indexed.groupby("label_name", sort=True).head(8)
    for row in sample.itertuples():
        prediction = predicted_labels[row.Index]
        predicted_label = id_to_label[prediction]
        result = "correct" if predicted_label == row.label_name else "wrong"
        text_path = cache_s3_object(client, row.asset_uri, CACHE_DIR / "texts")
        text = text_path.read_text(encoding="utf-8")
        lines.extend(
            [
                f"## {Path(row.key).stem}",
                "",
                f"Text: {text}",
                "",
                f"Actual intent: `{row.label_name}`",
                f"Predicted intent: `{predicted_label}`",
                f"Result: **{result}**",
                "",
            ]
        )
    write_markdown_report(LAB_DIR / "banking4_predictions.md", lines)


def save_embedding_map(client, validation_frame, model, tokenizer, max_tokens, device):
    sample = (
        validation_frame.reset_index(drop=True)
        .groupby("label_name", sort=True)
        .head(12)
        .reset_index(drop=True)
    )
    vectors = []
    labels = []

    model.eval()
    with torch.no_grad():
        for row in sample.itertuples(index=False):
            text_path = cache_s3_object(client, row.asset_uri, CACHE_DIR / "texts")
            text = text_path.read_text(encoding="utf-8")
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_tokens,
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in encoded.items()}
            output = model(**inputs, output_hidden_states=True)
            embedding = output.hidden_states[-1][:, 0, :].squeeze(0).cpu()
            vectors.append(embedding.numpy())
            labels.append(row.label_name)

    points = PCA(n_components=2).fit_transform(vectors)
    label_names = sorted(set(labels))
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 6))
    for index, label in enumerate(label_names):
        xs = [
            point[0] for point, row_label in zip(points, labels) if row_label == label
        ]
        ys = [
            point[1] for point, row_label in zip(points, labels) if row_label == label
        ]
        ax.scatter(
            xs,
            ys,
            label=label,
            s=42,
            alpha=0.82,
            color=colors[index % len(colors)],
        )
    ax.set_title("Banking77 Transformer Embedding Map")
    ax.set_xlabel("2D projection axis 1")
    ax.set_ylabel("2D projection axis 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(LAB_DIR / "banking5_embedding_map.png", dpi=160)
    plt.close(fig)


def cache_selected_texts(client, manifest) -> None:
    text_count = len(manifest)
    print(f"Downloading and caching {text_count:,} selected texts...")
    print("Already cached texts are skipped.")
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        cache_s3_object(client, row.asset_uri, CACHE_DIR / "texts")
        if index == 1 or index % 100 == 0 or index == text_count:
            print(f"Cached {index:,}/{text_count:,} texts", flush=True)
    print("Text cache is ready.")


def main() -> None:
    require_completed(str(MAX_TOKENS), "Fill in MAX_TOKENS before running this lab.")

    print("Reading Banking77 manifest from object storage...")
    client, _participant_bucket, shared_bucket = load_s3()
    manifest = read_manifest(client, shared_bucket, MANIFEST_KEY)
    require_columns(
        manifest,
        {"asset_uri", "key", "label_name", "split", "characters"},
        "Banking77 manifest",
    )
    manifest, label_names = selected_manifest(manifest)
    max_tokens = int(MAX_TOKENS)

    id_to_label = {index: label for index, label in enumerate(label_names)}
    label_to_id = {label: index for index, label in id_to_label.items()}

    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()

    print("Selected intent labels:")
    for label in label_names:
        print(f"- {label}")
    print(f"Training examples: {len(train_frame):,}")
    print(f"Validation examples: {len(validation_frame):,}")

    cache_selected_texts(client, manifest)
    save_class_distribution(
        manifest, LAB_DIR / "banking1_selected_class_distribution.png"
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading transformer model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    print("Building PyTorch datasets and dataloaders...")
    train_dataset = BankingTextDataset(train_frame, client, tokenizer, max_tokens)
    validation_dataset = BankingTextDataset(
        validation_frame, client, tokenizer, max_tokens
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, device
        )
        true_labels, predicted_labels, validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
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
    save_training_curve(history, LAB_DIR / "banking2_training_curve.png")
    save_confusion_matrix(
        matrix,
        label_names,
        LAB_DIR / "banking3_confusion_matrix.png",
        "Banking77 validation confusion matrix",
    )
    write_prediction_examples(client, validation_frame, predicted_labels, id_to_label)
    save_embedding_map(client, validation_frame, model, tokenizer, max_tokens, device)

    print(f"Final validation accuracy: {validation_accuracy:.3f}")
    print(f"Wrote outputs in {LAB_DIR}")


if __name__ == "__main__":
    main()

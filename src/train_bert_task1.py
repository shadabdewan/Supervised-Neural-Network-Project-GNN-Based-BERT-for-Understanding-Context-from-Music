"""Task 1 training pipeline for BERT genre classification.

This file contains the original training-specific logic that was extracted out of
bert_encoder.py. It imports the shared encoder and classifier from the slimmed
module and preserves the same output paths and behavior as the original script.
"""

import json
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from bert_encoder import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    BALANCED_CSV_PATH,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    GENRE2ID_PATH,
    GENRES,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_TOKEN_LENGTH,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    NUM_GENRES,
    OUTPUT_CSV_PATH,
    ROWS_PER_GENRE,
    BertGenreClassifier,
    tokenize_batch,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Task 1 helpers moved from bert_encoder.py
# ---------------------------------------------------------------------------
def extract_numeric_index(filename: str) -> int:
    """Extract the numeric index from a filename like 'pop.00042.wav'."""
    parts = filename.split(".")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return float("inf")
    return float("inf")


def select_balanced_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select the first ROWS_PER_GENRE valid rows per genre."""
    selected_rows = []

    for genre in GENRES:
        genre_df = df[df["genre"] == genre].copy()
        valid_mask = (
            (genre_df["lyrics_flag"] == False)
            & (genre_df["error"].isna() | (genre_df["error"] == ""))
            & (genre_df["transcription"].notna())
            & (genre_df["transcription"].str.strip() != "")
        )
        valid_df = genre_df[valid_mask].copy()
        valid_df["_idx"] = valid_df["filename"].apply(extract_numeric_index)
        valid_df = valid_df.sort_values("_idx").reset_index(drop=True)
        valid_df = valid_df.drop("_idx", axis=1)

        selected = valid_df.head(ROWS_PER_GENRE)
        selected_rows.append(selected)

        if len(selected) < ROWS_PER_GENRE:
            print(
                f"Warning: only {len(selected)} valid rows found for genre '{genre}', "
                f"expected {ROWS_PER_GENRE}"
            )

    return pd.concat(selected_rows, ignore_index=True)


class LyricsDataset(Dataset):
    """PyTorch Dataset for lyrics and genre labels."""

    def __init__(self, transcriptions: List[str], genre_ids: List[int], tokenizer, max_length: int = MAX_TOKEN_LENGTH):
        self.transcriptions = transcriptions
        self.genre_ids = genre_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx: int):
        text = self.transcriptions[idx]
        genre_id = self.genre_ids[idx]

        encoding = tokenize_batch([text], self.tokenizer, max_length=self.max_length)
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        label = np.zeros(NUM_GENRES, dtype=np.float32)
        label[genre_id] = 1.0
        label = torch.tensor(label, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


def compute_token_length_stats(df: pd.DataFrame, tokenizer) -> None:
    """Compute and print token length statistics."""
    token_lengths = []

    for text in df["transcription"]:
        tokens = tokenizer.encode(text, max_length=10000, truncation=False)
        token_lengths.append(len(tokens))

    token_lengths = np.array(token_lengths)
    p50 = np.percentile(token_lengths, 50)
    p90 = np.percentile(token_lengths, 90)
    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)
    max_len = np.max(token_lengths)
    exceed_256 = np.sum(token_lengths > MAX_TOKEN_LENGTH)
    exceed_pct = 100 * exceed_256 / len(token_lengths)

    print("\n" + "=" * 70)
    print("Token Length Distribution")
    print("=" * 70)
    print(f"50th percentile: {p50:.0f}")
    print(f"90th percentile: {p90:.0f}")
    print(f"95th percentile: {p95:.0f}")
    print(f"99th percentile: {p99:.0f}")
    print(f"Maximum: {max_len:.0f}")
    print(f"Rows exceeding {MAX_TOKEN_LENGTH} tokens: {exceed_256}/{len(token_lengths)} ({exceed_pct:.2f}%)")
    print("=" * 70 + "\n")


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = torch.argmax(probs, dim=1)
            true_labels = torch.argmax(labels, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    avg_val_loss = total_loss / len(val_loader)

    return accuracy, macro_f1, micro_f1, all_preds, all_labels, avg_val_loss


def predict_with_probs(model, val_loader, device):
    """Return predictions, probabilities, and true labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = torch.argmax(probs, dim=1)
            true_labels = torch.argmax(labels, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def plot_f1_curves(train_macro, train_micro, val_macro, val_micro):
    """Plot Macro-F1 and Micro-F1 curves across epochs."""
    epochs = range(1, len(train_macro) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_macro, "o-", label="Train Macro-F1", linewidth=2)
    plt.plot(epochs, val_macro, "s-", label="Val Macro-F1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_micro, "o-", label="Train Micro-F1", linewidth=2)
    plt.plot(epochs, val_micro, "s-", label="Val Micro-F1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Micro-F1")
    plt.title("Micro-F1 across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "data" / "processed" / "f1_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    print(f"F1 curves saved to {plot_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_GENRES))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GENRES,
        yticklabels=GENRES,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.title("Confusion Matrix - Genre Classification")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = PROJECT_ROOT / "data" / "processed" / "confusion_matrix.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    print(f"Confusion matrix saved to {plot_path}")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("BERT Genre Classification - Main Pipeline")
    print("=" * 70 + "\n")

    print("Step 1: Loading and selecting balanced rows...")
    if not OUTPUT_CSV_PATH.exists():
        print(f"Error: {OUTPUT_CSV_PATH} not found!")
        return

    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"Loaded {len(df)} total rows from {OUTPUT_CSV_PATH}")

    balanced_df = select_balanced_rows(df)
    print(f"Selected {len(balanced_df)} balanced rows (genre distribution below)")
    print(balanced_df["genre"].value_counts().sort_index())

    balanced_df.to_csv(BALANCED_CSV_PATH, index=False)
    print(f"Saved balanced dataset to {BALANCED_CSV_PATH}\n")

    print("Step 2: Creating genre2id mapping...")
    genre2id = {genre: idx for idx, genre in enumerate(GENRES)}
    id2genre = {idx: genre for genre, idx in genre2id.items()}

    with open(GENRE2ID_PATH, "w") as f:
        json.dump(genre2id, f, indent=2)
    print(f"Saved genre2id mapping to {GENRE2ID_PATH}")
    print(f"Mapping: {genre2id}\n")

    print("Step 3: Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    compute_token_length_stats(balanced_df, tokenizer)

    print("Step 4: Preparing labels...")
    genre_ids = [genre2id[g] for g in balanced_df["genre"]]
    transcriptions = balanced_df["transcription"].tolist()

    print("Step 5: Splitting data (stratified by genre)...")
    train_val_indices, test_indices, _, _ = train_test_split(
        range(len(transcriptions)),
        genre_ids,
        test_size=0.15,
        stratify=genre_ids,
        random_state=42,
    )

    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices,
        [genre_ids[i] for i in train_val_indices],
        test_size=0.176,
        stratify=[genre_ids[i] for i in train_val_indices],
        random_state=42,
    )

    train_transcriptions = [transcriptions[i] for i in train_indices]
    train_genre_ids = [genre_ids[i] for i in train_indices]
    val_transcriptions = [transcriptions[i] for i in val_indices]
    val_genre_ids = [genre_ids[i] for i in val_indices]
    test_transcriptions = [transcriptions[i] for i in test_indices]
    test_genre_ids = [genre_ids[i] for i in test_indices]

    print(f"Train set: {len(train_indices)} samples")
    print(f"Val set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")

    print("\nGenre distribution in splits:")
    print("Train:", dict(pd.Series(train_genre_ids).apply(lambda x: id2genre[x]).value_counts().sort_index()))
    print("Val:", dict(pd.Series(val_genre_ids).apply(lambda x: id2genre[x]).value_counts().sort_index()))
    print("Test:", dict(pd.Series(test_genre_ids).apply(lambda x: id2genre[x]).value_counts().sort_index()))
    print()

    print("Step 6: Creating datasets and dataloaders...")
    train_dataset = LyricsDataset(train_transcriptions, train_genre_ids, tokenizer)
    val_dataset = LyricsDataset(val_transcriptions, val_genre_ids, tokenizer)
    test_dataset = LyricsDataset(test_transcriptions, test_genre_ids, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Created dataloaders (batch size: {BATCH_SIZE})\n")

    print("Step 7: Initializing model...")
    model = BertGenreClassifier(num_genres=NUM_GENRES).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Model architecture: BertGenreClassifier")
    print(f"Optimizer: AdamW (lr={LEARNING_RATE})")
    print(f"Loss: BCEWithLogitsLoss")
    print(f"\n--- Training Schedule ---")
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs\n")

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    train_macro_f1_list = []
    train_micro_f1_list = []
    val_macro_f1_list = []
    val_micro_f1_list = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE)

        train_acc, train_macro, train_micro, _, _, _ = evaluate(model, train_loader, DEVICE)
        train_macro_f1_list.append(train_macro)
        train_micro_f1_list.append(train_micro)

        val_acc, val_macro, val_micro, _, _, val_loss = evaluate(model, val_loader, DEVICE)
        val_macro_f1_list.append(val_macro)
        val_micro_f1_list.append(val_micro)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro-F1: {train_macro:.4f} | Train Micro-F1: {train_micro:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro:.4f} | Val Micro-F1: {val_micro:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Validation loss improved. Best model saved to {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n✗ Early stopping triggered after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("\n" + "=" * 70)
    print(f"Training completed! Best model found at epoch {best_epoch}")
    print("=" * 70 + "\n")

    print("Step 8: Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Loaded best model from {BEST_MODEL_PATH}\n")

    test_acc, test_macro_f1, test_micro_f1, test_preds, test_labels, _ = evaluate(model, test_loader, DEVICE)

    print("Test Set Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro-F1: {test_macro_f1:.4f}")
    print(f"  Micro-F1: {test_micro_f1:.4f}\n")

    print("Step 9: Plotting results...")
    plot_f1_curves(train_macro_f1_list, train_micro_f1_list, val_macro_f1_list, val_micro_f1_list)
    plot_confusion_matrix(test_labels, test_preds)

    print("Step 10: Example predictions from test set...\n")
    test_preds, test_probs, _ = predict_with_probs(model, test_loader, DEVICE)

    num_examples = min(5, len(test_transcriptions))
    for i in range(num_examples):
        lyrics_snippet = test_transcriptions[i][:200]
        true_genre = id2genre[test_genre_ids[i]]
        pred_genre = id2genre[test_preds[i]]
        pred_probs = test_probs[i]
        top3_indices = np.argsort(pred_probs)[-3:][::-1]
        top3_probs = [(id2genre[idx], pred_probs[idx]) for idx in top3_indices]

        print(f"Example {i + 1}:")
        print(f"  Lyrics: '{lyrics_snippet}...'")
        print(f"  True genre: {true_genre}")
        print(f"  Predicted genre: {pred_genre}")
        print(f"  Top-3 predictions: {[f'{g}({p:.3f})' for g, p in top3_probs]}")
        print()

    print("Step 11: Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}\n")

    print("=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

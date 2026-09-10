"""
Shared BERT encoder used by both Task 1 and Task 3.
This file is now the shared encoder used by both Task 1 (via BertGenreClassifier)
and Task 3 (via BertLyricsEncoder directly).
"""

import warnings
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared configuration used by Task 1 and Task 3 callers
# ---------------------------------------------------------------------------
GENRES = ["blues", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"]
ROWS_PER_GENRE = 80
MAX_TOKEN_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
HIDDEN_SIZE = 768
NUM_GENRES = len(GENRES)
EARLY_STOPPING_PATIENCE = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics_transcriptions.csv"
BALANCED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics_transcriptions_balanced.csv"
GENRE2ID_PATH = PROJECT_ROOT / "data" / "processed" / "genre2id.json"
MODEL_SAVE_PATH = PROJECT_ROOT / "data" / "processed" / "bert_genre_model.pt"
BEST_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "bert_genre_model_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertLyricsEncoder(nn.Module):
    """Reusable BERT encoder returning CLS embedding and full token-level output."""

    def __init__(self, model_name: str = "bert-base-uncased", freeze_bert: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """Return (cls_embedding, last_hidden_state)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding, outputs.last_hidden_state


def tokenize_batch(
    texts: Sequence[str],
    tokenizer=None,
    max_length: int = MAX_TOKEN_LENGTH,
    padding: str = "max_length",
    truncation: bool = True,
):
    """Tokenize a list of texts with the project's original settings."""
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        list(texts),
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors="pt",
    )


class BertGenreClassifier(nn.Module):
    """Backward-compatible Task 1 classifier built on top of the shared encoder."""

    def __init__(self, num_genres: int = NUM_GENRES):
        super().__init__()
        self.encoder = BertLyricsEncoder()
        self.linear = nn.Linear(HIDDEN_SIZE, num_genres)

    def forward(self, input_ids, attention_mask):
        cls_embedding, _ = self.encoder(input_ids, attention_mask)
        logits = self.linear(cls_embedding)
        return logits

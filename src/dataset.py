"""
dataset.py
----------
DailyDialog loading, parsing, and PyTorch Dataset/DataLoader utilities
for emotion-shift modeling in dyadic dialogues.

Paper: "Speaker-State Memory for Emotion-Shift Modeling in Dyadic Dialogues"
Author: Tulika Sharma (stulika029@gmail.com)
"""

import os
import re
import ast
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

PAD_EMOTION_ID = -100
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_MAP = {i: label for i, label in enumerate(EMOTION_LABELS)}


# ─── Parsing Utilities ────────────────────────────────────────────────────────

def parse_dialog_field(s: str) -> list:
    """Parse the raw dialog string from DailyDialog CSV into a list of utterances."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    parts = re.split(r"'\s*'\s*", s)
    parts = [p.strip().strip("'\"") for p in parts if p.strip()]
    return parts


def parse_emotion_field(s: str) -> list:
    """Parse the raw emotion string from DailyDialog CSV into a list of integer IDs."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    parts = s.replace(",", " ").split()
    nums = []
    for x in parts:
        try:
            nums.append(int(x))
        except ValueError:
            continue
    return nums


def fix_list_format(s) -> list:
    """Fix malformed list strings like '[3 4 12 0]' → [3, 4, 12, 0]."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        inner = re.sub(r"\s+", ",", inner)
        fixed = f"[{inner}]"
        try:
            val = ast.literal_eval(fixed)
            if isinstance(val, list):
                return val
        except Exception:
            return None
    return None


def load_and_clean_split(data_dir: str, split: str) -> pd.DataFrame:
    """
    Load and clean a DailyDialog CSV split.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing {split}.csv files.
    split : str
        One of 'train', 'validation', 'test'.

    Returns
    -------
    pd.DataFrame with columns: dialog (list), act, emotion (list of int)
    """
    file_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

    print(f"Loaded {file_path} with {len(df)} rows.")

    cleaned = []
    dropped = 0

    for _, row in df.iterrows():
        parsed_dialog = parse_dialog_field(str(row.get("dialog", "")))
        parsed_emotion = parse_emotion_field(str(row.get("emotion", "")))

        if not isinstance(parsed_dialog, list) or not isinstance(parsed_emotion, list):
            dropped += 1
            continue

        min_len = min(len(parsed_dialog), len(parsed_emotion))
        if min_len == 0:
            dropped += 1
            continue

        cleaned.append({
            "dialog": parsed_dialog[:min_len],
            "act": row.get("act", None),
            "emotion": parsed_emotion[:min_len],
        })

    print(f"Kept {len(cleaned)} valid dialogues ({dropped} dropped) from split '{split}'.")
    return pd.DataFrame(cleaned) if cleaned else pd.DataFrame()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DailyDialogLocalDataset(Dataset):
    """
    PyTorch Dataset wrapping DailyDialog CSV files.

    Produces per-turn examples with:
    - context: last k_context turns (padded left)
    - current_turn: the turn being predicted
    - shift_label: 1 if speaker's emotion changed since their last turn, else 0
                   (-100 = masked; first occurrence of a speaker has no prior)
    - has_prev: bool, whether a prior speaker turn exists for shift calculation
    - prev_emotion_id: emotion at previous speaker turn (-100 if none)

    Parameters
    ----------
    data_dir : str
        Directory containing train.csv / validation.csv / test.csv.
    split : str
        One of 'train', 'validation', 'test'.
    k_context : int
        Number of prior turns to include as context window (default: 4).
    """

    def __init__(self, data_dir: str, split: str = "train", k_context: int = 4):
        self.data_dir = data_dir
        self.split = split
        self.k_context = k_context

        self.data = load_and_clean_split(data_dir, split)
        if self.data.empty:
            raise ValueError(f"No valid data found for split '{split}' in {data_dir}")

        self.examples = []
        valid_dialogs = 0

        for dialog_id, row in self.data.iterrows():
            utterances = row["dialog"]
            emotions = row["emotion"]

            if not isinstance(utterances, list) or not isinstance(emotions, list):
                continue

            min_len = min(len(utterances), len(emotions))
            if min_len == 0:
                continue

            utterances = utterances[:min_len]
            emotions = [int(e) for e in emotions[:min_len]]
            valid_dialogs += 1

            # Build turn list; speakers alternate A/B
            turns = [
                {
                    "speaker": "A" if i % 2 == 0 else "B",
                    "text": utterances[i],
                    "emotion_id": emotions[i],
                    "turn_idx": i,
                }
                for i in range(min_len)
            ]

            last_emotion_by_speaker = {"A": None, "B": None}

            for i, turn in enumerate(turns):
                speaker = turn["speaker"]
                current_emotion = turn["emotion_id"]
                has_prev = last_emotion_by_speaker[speaker] is not None

                if has_prev:
                    prev_emotion = last_emotion_by_speaker[speaker]
                    shift_label = int(current_emotion != prev_emotion)
                    prev_emotion_id = prev_emotion
                else:
                    shift_label = PAD_EMOTION_ID
                    prev_emotion_id = PAD_EMOTION_ID

                last_emotion_by_speaker[speaker] = current_emotion

                # Build left-padded context
                context_start = max(0, i - k_context)
                context_turns = turns[context_start:i]
                while len(context_turns) < k_context:
                    context_turns = [{"speaker": "PAD", "text": "", "emotion_id": PAD_EMOTION_ID, "turn_idx": -1}] + context_turns

                self.examples.append({
                    "dialog_id": dialog_id,
                    "turn_idx": i,
                    "context": context_turns,
                    "current_turn": turn,
                    "has_prev": has_prev,
                    "prev_emotion_id": prev_emotion_id,
                    "shift_label": shift_label,
                })

        print(f"  Valid dialogues: {valid_dialogs} | Total turn examples: {len(self.examples)}")
        if not self.examples:
            raise ValueError("No valid examples built from dataset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─── Collate Functions ────────────────────────────────────────────────────────

def collate_fn_b1(batch, tokenizer, max_len: int = 128) -> dict:
    """
    Collate function for B1 (utterance-only model).
    Encodes only the current utterance — no context.
    """
    current_turns = [item["current_turn"] for item in batch]
    texts = [f"{t['speaker']}: {t['text']}" for t in current_turns]

    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    return {
        "current_input_ids": enc["input_ids"],
        "current_attention_mask": enc["attention_mask"],
        "shift_labels": torch.tensor([item["shift_label"] for item in batch], dtype=torch.long),
        "emotion_labels": torch.tensor([t["emotion_id"] for t in current_turns], dtype=torch.long),
        "has_prev_mask": torch.tensor([item["has_prev"] for item in batch], dtype=torch.bool),
        "prev_emotion_ids": torch.tensor([item["prev_emotion_id"] for item in batch], dtype=torch.long),
        "dialog_ids": [item["dialog_id"] for item in batch],
        "turn_indices": [item["turn_idx"] for item in batch],
    }


def collate_fn_b2(batch, tokenizer, max_len: int = 512) -> dict:
    """
    Collate function for B2 (contextual model).
    Encodes K=4 prior turns concatenated with [SPK_A]/[SPK_B] tags and [SEP].
    """
    contexts = [item["context"] for item in batch]
    current_turns = [item["current_turn"] for item in batch]

    context_strings = []
    for ctx in contexts:
        parts = []
        for turn in ctx:
            if turn["speaker"] == "PAD":
                parts.append("")
            else:
                tag = "[SPK_A]" if turn["speaker"] == "A" else "[SPK_B]"
                parts.append(f"{tag} {turn['text']}")
        context_strings.append(" [SEP] ".join(parts))

    enc = tokenizer(context_strings, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    return {
        "context_input_ids": enc["input_ids"],
        "context_attention_mask": enc["attention_mask"],
        "shift_labels": torch.tensor([item["shift_label"] for item in batch], dtype=torch.long),
        "emotion_labels": torch.tensor([t["emotion_id"] for t in current_turns], dtype=torch.long),
        "has_prev_mask": torch.tensor([item["has_prev"] for item in batch], dtype=torch.bool),
        "prev_emotion_ids": torch.tensor([item["prev_emotion_id"] for item in batch], dtype=torch.long),
        "dialog_ids": [item["dialog_id"] for item in batch],
        "turn_indices": [item["turn_idx"] for item in batch],
    }


def collate_fn_b3(batch, tokenizer, max_len: int = 512) -> dict:
    """
    Collate function for B3 (memory model).
    Encodes context with [SPK_A]/[SPK_B]/[SEP] tokens and also returns
    speaker IDs, dialog IDs, and raw texts for memory routing.
    """
    contexts = [item["context"] for item in batch]
    current_turns = [item["current_turn"] for item in batch]

    context_strings = []
    for ctx in contexts:
        parts = []
        for turn in ctx:
            if turn["speaker"] == "PAD":
                parts.append("")
            else:
                tag = "[SPK_A]" if turn["speaker"] == "A" else "[SPK_B]"
                parts.append(f"{tag} {turn['text']}")
        context_strings.append(" [SEP] ".join(parts))

    enc = tokenizer(context_strings, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "speaker_ids": [t["speaker"] for t in current_turns],
        "dialog_ids": [item["dialog_id"] for item in batch],
        "turn_indices": [item["turn_idx"] for item in batch],
        "texts": [t["text"] for t in current_turns],
        "shift_labels": torch.tensor([item["shift_label"] for item in batch], dtype=torch.long),
        "emotion_labels": torch.tensor([t["emotion_id"] for t in current_turns], dtype=torch.long),
        "has_prev_mask": torch.tensor([item["has_prev"] for item in batch], dtype=torch.bool),
        "prev_emotion_ids": torch.tensor([item["prev_emotion_id"] for item in batch], dtype=torch.long),
    }


# ─── Class Weight Utility ─────────────────────────────────────────────────────

def calculate_class_weights(dataset: DailyDialogLocalDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for the binary shift label
    to counteract extreme class imbalance (most turns have no shift).

    Returns
    -------
    torch.Tensor of shape (2,) — [weight_no_shift, weight_shift]
    """
    labels = [
        int(item["shift_label"])
        for item in dataset
        if item["has_prev"] and item["shift_label"] != PAD_EMOTION_ID
    ]
    if not labels:
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    labels = np.array(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    counts = np.where(counts == 0, 1, counts)
    total = labels.shape[0]
    weights = total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float)

"""
models.py
---------
Model definitions for emotion-shift modeling in dyadic dialogues:

  B0  MajorityBaseline         — always predict majority class / no shift
  B1  UtteranceOnlyModel       — RoBERTa on current utterance only
  B2  ContextualModel          — RoBERTa on K=4 context window
  B3  MemoryAugmentedModel     — Speaker-state memory (GRU or Gated Residual)
  B3  GRUMemoryWithSentiment   — Speaker-state memory + sentiment cue (best)

Paper: "Speaker-State Memory for Emotion-Shift Modeling in Dyadic Dialogues"
Author: Tulika Sharma (stulika029@gmail.com)
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel


# ─── Base Model ───────────────────────────────────────────────────────────────

class BaseRobertaEmotionShift(nn.Module):
    """
    Shared backbone: RoBERTa encoder + emotion head + shift head.

    Parameters
    ----------
    num_emotions : int
        Number of emotion classes (7 for DailyDialog).
    lambda_shift : float
        Weight for the shift-detection loss term (default 0.5).
    tokenizer : optional
        If provided, resizes token embeddings to match tokenizer length.
    """

    def __init__(self, num_emotions: int = 7, lambda_shift: float = 0.5, tokenizer=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        if tokenizer is not None:
            self.roberta.resize_token_embeddings(len(tokenizer))

        self.emotion_head = nn.Linear(768, num_emotions)
        self.shift_head   = nn.Linear(768, 1)
        self.lambda_shift = lambda_shift
        self.dropout      = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask,
                emotion_labels=None, shift_labels=None,
                has_prev_mask=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.dropout(outputs.pooler_output)

        emotion_logits = self.emotion_head(pooled)
        shift_logits   = self.shift_head(pooled).squeeze(-1)

        loss = 0
        if emotion_labels is not None:
            loss += nn.CrossEntropyLoss()(emotion_logits, emotion_labels)

        if shift_labels is not None and has_prev_mask is not None:
            valid = has_prev_mask & (shift_labels != -100)
            if valid.any():
                loss += self.lambda_shift * nn.BCEWithLogitsLoss()(
                    shift_logits[valid], shift_labels[valid].float()
                )

        return {"emotion_logits": emotion_logits, "shift_logits": shift_logits, "loss": loss}


# ─── Baselines ────────────────────────────────────────────────────────────────

class B0MajorityBaseline:
    """
    B0: Majority class baseline.
    Always predicts the most frequent emotion and no shift.

    Not a nn.Module — use .fit() on a dataset, then .predict() on batches.
    """

    def __init__(self):
        self.majority_emotion = None

    def fit(self, dataset) -> None:
        """Compute majority emotion from the training dataset."""
        all_emotions = [
            item["current_turn"]["emotion_id"]
            for item in dataset
            if item["current_turn"]["emotion_id"] != -100
        ]
        counts = np.bincount(all_emotions)
        self.majority_emotion = int(np.argmax(counts))

    def predict(self, batch: dict) -> dict:
        """Return majority-class emotion predictions and zero shift predictions."""
        n = len(batch["emotion_labels"])
        return {
            "emotion_preds": torch.full((n,), self.majority_emotion, dtype=torch.long),
            "shift_preds":   torch.zeros(n),
        }


class B1UtteranceOnlyModel(BaseRobertaEmotionShift):
    """
    B1: Single utterance model — no context window.
    Encodes only the current turn with RoBERTa.
    """

    def forward(self, current_input_ids, current_attention_mask, **kwargs):
        return super().forward(current_input_ids, current_attention_mask, **kwargs)


class B2ContextualModel(BaseRobertaEmotionShift):
    """
    B2: Contextual model — encodes the last K=4 turns concatenated with
    [SPK_A]/[SPK_B] and [SEP] tokens.
    """

    def forward(self, context_input_ids, context_attention_mask, **kwargs):
        return super().forward(context_input_ids, context_attention_mask, **kwargs)


# ─── Memory-Augmented Models ──────────────────────────────────────────────────

class MemoryAugmentedModel(BaseRobertaEmotionShift):
    """
    B3: Speaker-State Memory Model.

    Maintains a per-(dialog, speaker) latent state that is updated each turn.
    The fused representation z_t = [h_t; s_t; h_ctx] is used for both heads.

    Parameters
    ----------
    memory_type : str
        'GRU' (default) or 'Gated' (gated residual update).
    memory_dim : int
        Dimensionality of the speaker state vector (default 128).
    use_sentiment : bool
        If True, append VADER-style sentiment features to h_t before GRU update.
    truncate_memory : int or None
        If set, keeps only the last N memory history entries per speaker.
    """

    def __init__(self, num_emotions: int = 7, lambda_shift: float = 0.5,
                 memory_type: str = "GRU", memory_dim: int = 128,
                 use_sentiment: bool = False, truncate_memory=None):
        super().__init__(num_emotions, lambda_shift)

        self.memory_type     = memory_type
        self.memory_dim      = memory_dim
        self.use_sentiment   = use_sentiment
        self.truncate_memory = truncate_memory

        memory_input_dim = 768 + (3 if use_sentiment else 0)

        if memory_type == "GRU":
            self.gru_cell = nn.GRUCell(memory_input_dim, memory_dim)
        elif memory_type == "Gated":
            self.gate_w = nn.Linear(memory_input_dim + memory_dim, memory_dim)
            self.gate_v = nn.Linear(memory_input_dim, memory_dim)

        # Override heads to use fused dim: h_t + s_t + h_ctx
        fusion_dim = 768 + memory_dim + 768
        self.emotion_head = nn.Linear(fusion_dim, num_emotions)
        self.shift_head   = nn.Linear(fusion_dim, 1)

        self.speaker_memory:  dict = {}
        self.memory_history:  dict = {}

    # ── Memory Management ──

    def reset_memory(self, dialog_id=None) -> None:
        """Reset memory for a specific dialogue or all dialogues."""
        if dialog_id is None:
            self.speaker_memory = {}
            self.memory_history = {}
        else:
            keys = [k for k in self.speaker_memory if k[0] == dialog_id]
            for k in keys:
                del self.speaker_memory[k]
                self.memory_history.pop(k, None)

    def get_speaker_memory(self, dialog_id, speaker) -> torch.Tensor:
        key = (dialog_id, speaker)
        if key not in self.speaker_memory:
            self.speaker_memory[key] = torch.zeros(
                self.memory_dim, device=next(self.parameters()).device
            )
            self.memory_history[key] = []
        return self.speaker_memory[key]

    def update_speaker_memory(self, dialog_id, speaker, new_memory: torch.Tensor) -> None:
        key = (dialog_id, speaker)
        self.speaker_memory[key] = new_memory
        if key in self.memory_history:
            self.memory_history[key].append(new_memory.detach().cpu().numpy())
            if self.truncate_memory and len(self.memory_history[key]) > self.truncate_memory:
                self.memory_history[key] = self.memory_history[key][-self.truncate_memory:]

    # ── Sentiment Features ──

    def extract_sentiment_features(self, texts: list) -> torch.Tensor:
        """
        Rule-based sentiment feature extraction (positive / neutral / negative).
        Replace with VADER or a learned scorer for production use.
        """
        POS = {"good", "great", "excellent", "happy", "love", "nice", "awesome"}
        NEG = {"bad", "terrible", "hate", "awful", "horrible", "sad"}
        sentiments = []
        for text in texts:
            words = text.lower().split()
            pos = sum(1 for w in words if w in POS)
            neg = sum(1 for w in words if w in NEG)
            neu = max(1, 10 - pos - neg)
            total = pos + neg + neu + 1e-8
            sentiments.append([pos / total, neu / total, neg / total])
        return torch.tensor(sentiments, device=next(self.parameters()).device)

    # ── Forward ──

    def forward(self, input_ids, attention_mask, speaker_ids, dialog_ids,
                emotion_labels=None, shift_labels=None, has_prev_mask=None,
                texts=None, **kwargs):

        batch_size = input_ids.size(0)
        device     = input_ids.device

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h_t   = outputs.pooler_output              # [B, 768]
        h_ctx = outputs.last_hidden_state.mean(1)  # [B, 768]

        if self.use_sentiment and texts is not None:
            sentiment_features = self.extract_sentiment_features(texts)
            h_t = torch.cat([h_t, sentiment_features], dim=1)

        memory_states    = []
        updated_memories = []

        for i in range(batch_size):
            s_t = self.get_speaker_memory(dialog_ids[i], speaker_ids[i])

            if self.memory_type == "GRU":
                s_new = self.gru_cell(h_t[i].unsqueeze(0), s_t.unsqueeze(0)).squeeze(0)
            elif self.memory_type == "Gated":
                concat = torch.cat([h_t[i], s_t], dim=0)
                gate   = torch.sigmoid(self.gate_w(concat))
                update = torch.tanh(self.gate_v(h_t[i]))
                s_new  = s_t + gate * update
            else:
                s_new = s_t

            memory_states.append(s_t)
            updated_memories.append((dialog_ids[i], speaker_ids[i], s_new))

        s_t_batch = torch.stack(memory_states) if memory_states else torch.zeros(batch_size, self.memory_dim, device=device)
        z_t = self.dropout(torch.cat([h_t, s_t_batch, h_ctx], dim=1))

        emotion_logits = self.emotion_head(z_t)
        shift_logits   = self.shift_head(z_t).squeeze(-1)

        # Commit memory updates
        for d_id, spk, s_new in updated_memories:
            self.update_speaker_memory(d_id, spk, s_new)

        loss = 0
        if emotion_labels is not None:
            loss += nn.CrossEntropyLoss()(emotion_logits, emotion_labels)
        if shift_labels is not None and has_prev_mask is not None:
            valid = has_prev_mask & (shift_labels != -100)
            if valid.any():
                loss += self.lambda_shift * nn.BCEWithLogitsLoss()(
                    shift_logits[valid], shift_labels[valid].float()
                )

        return {
            "emotion_logits": emotion_logits,
            "shift_logits":   shift_logits,
            "loss":           loss,
            "memory_states":  s_t_batch,
        }


class GRUMemoryWithSentimentModel(BaseRobertaEmotionShift):
    """
    B3 (Best): GRU Speaker-State Memory with Sentiment Features.

    Concatenates VADER-style sentiment features [pos, neu, neg] to the
    RoBERTa pooled output before the GRU update, giving the memory an
    explicit polarity signal. This variant achieves the highest shift F1.

    Architecture:
        h_t_aug = [h_t; r_t]                  (768 + 3 = 771)
        s_t^i   = GRU(h_t_aug, s_{t-1}^i)    memory update
        z_t     = [h_t_aug; s_t^i; h_ctx]    fused rep for prediction
        e_t     = softmax(W_e · z_t)          emotion head
        y_t     = sigmoid(w_s · z_t)          shift head

    Parameters
    ----------
    memory_dim : int
        Dimension of speaker latent state (default 128).
    truncate_memory : int or None
        Limit memory history length per speaker.
    """

    def __init__(self, num_emotions: int = 7, lambda_shift: float = 0.5,
                 memory_dim: int = 128, truncate_memory=None):
        super().__init__(num_emotions, lambda_shift)

        self.memory_dim      = memory_dim
        self.truncate_memory = truncate_memory

        gru_input_dim  = 768 + 3   # RoBERTa + 3 sentiment features
        fusion_dim     = gru_input_dim + memory_dim + 768

        self.gru_cell     = nn.GRUCell(gru_input_dim, memory_dim)
        self.emotion_head = nn.Linear(fusion_dim, num_emotions)
        self.shift_head   = nn.Linear(fusion_dim, 1)

        nn.init.xavier_uniform_(self.emotion_head.weight)
        nn.init.xavier_uniform_(self.shift_head.weight)

        self.speaker_memory: dict = {}
        self.memory_history: dict = {}

    # ── Memory Management ──

    def reset_memory(self, dialog_id=None) -> None:
        if dialog_id is None:
            self.speaker_memory = {}
            self.memory_history = {}
        else:
            keys = [k for k in self.speaker_memory if k[0] == dialog_id]
            for k in keys:
                del self.speaker_memory[k]
                self.memory_history.pop(k, None)

    def get_speaker_memory(self, dialog_id, speaker) -> torch.Tensor:
        key = (dialog_id, speaker)
        if key not in self.speaker_memory:
            self.speaker_memory[key] = torch.zeros(
                self.memory_dim, device=next(self.parameters()).device
            )
            self.memory_history[key] = []
        return self.speaker_memory[key]

    def update_speaker_memory(self, dialog_id, speaker, new_memory: torch.Tensor) -> None:
        key = (dialog_id, speaker)
        self.speaker_memory[key] = new_memory
        if key in self.memory_history:
            self.memory_history[key].append(new_memory.detach().cpu().numpy())
            if self.truncate_memory and len(self.memory_history[key]) > self.truncate_memory:
                self.memory_history[key] = self.memory_history[key][-self.truncate_memory:]

    # ── Sentiment Features ──

    def extract_sentiment_features(self, texts: list) -> torch.Tensor:
        """Rule-based positive / neutral / negative polarity features."""
        POS = {"good", "great", "excellent", "happy", "love", "nice",
               "awesome", "amazing", "wonderful"}
        NEG = {"bad", "terrible", "hate", "awful", "horrible", "sad",
               "angry", "disgusting"}
        sentiments = []
        for text in texts:
            words = text.lower().split()
            pos = sum(1 for w in words if w in POS)
            neg = sum(1 for w in words if w in NEG)
            neu = max(1, 10 - pos - neg)
            total = pos + neg + neu + 1e-8
            sentiments.append([pos / total, neu / total, neg / total])
        return torch.tensor(sentiments, device=next(self.parameters()).device)

    # ── Forward ──

    def forward(self, input_ids, attention_mask, speaker_ids, dialog_ids,
                emotion_labels=None, shift_labels=None, has_prev_mask=None,
                texts=None, **kwargs):

        batch_size = input_ids.size(0)
        device     = input_ids.device

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h_t   = outputs.pooler_output              # [B, 768]
        h_ctx = outputs.last_hidden_state.mean(1)  # [B, 768]

        sentiment = self.extract_sentiment_features(texts or [""] * batch_size)
        h_t_aug   = torch.cat([h_t, sentiment], dim=1)  # [B, 771]

        memory_states    = []
        updated_memories = []

        for i in range(batch_size):
            s_t = self.get_speaker_memory(dialog_ids[i], speaker_ids[i]).to(device)
            s_new = self.gru_cell(h_t_aug[i].unsqueeze(0), s_t.unsqueeze(0)).squeeze(0)
            memory_states.append(s_t)
            updated_memories.append((dialog_ids[i], speaker_ids[i], s_new))

        s_t_batch = torch.stack(memory_states) if memory_states else torch.zeros(batch_size, self.memory_dim, device=device)
        z_t = self.dropout(torch.cat([h_t_aug, s_t_batch, h_ctx], dim=1))

        emotion_logits = self.emotion_head(z_t)
        shift_logits   = self.shift_head(z_t).squeeze(-1)

        for d_id, spk, s_new in updated_memories:
            self.update_speaker_memory(d_id, spk, s_new)

        loss = 0
        if emotion_labels is not None:
            loss += nn.CrossEntropyLoss()(emotion_logits, emotion_labels)
        if shift_labels is not None and has_prev_mask is not None:
            valid = has_prev_mask & (shift_labels != -100)
            if valid.any():
                loss += self.lambda_shift * nn.BCEWithLogitsLoss()(
                    shift_logits[valid], shift_labels[valid].float()
                )

        return {
            "emotion_logits": emotion_logits,
            "shift_logits":   shift_logits,
            "loss":           loss,
        }

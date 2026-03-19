"""
train.py
--------
ExperimentRunner and EnhancedExperimentRunner:
training loops, evaluation, and per-turn metric collection
for emotion-shift modeling in dyadic dialogues.

Paper: "Speaker-State Memory for Emotion-Shift Modeling in Dyadic Dialogues"
Author: Tulika Sharma (stulika029@gmail.com)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm


# ─── Base Experiment Runner ───────────────────────────────────────────────────

class ExperimentRunner:
    """
    Standard training and evaluation loop.

    Parameters
    ----------
    model : nn.Module
        Any B1 / B2 / B3 model variant.
    train_loader, val_loader, test_loader : DataLoader
    device : torch.device
    """

    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device
        self.optimizer    = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ── Training ──

    def train_epoch(self) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            tensor_batch = self._to_device(batch)

            if hasattr(self.model, "reset_memory"):
                self.model.reset_memory()

            self.optimizer.zero_grad()
            outputs = self.model(**tensor_batch)
            loss    = outputs["loss"]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ── Evaluation ──

    def evaluate(self, loader, split_name: str = "Validation") -> dict:
        """
        Evaluate on a DataLoader split.

        Returns a dict with:
            emotion_macro_f1, shift_f1, shift_auprc,
            emotion_preds, emotion_labels,
            shift_preds, shift_labels, shift_probs,
            valid_shift_mask
        """
        self.model.eval()
        all_emotion_preds  = []
        all_emotion_labels = []
        all_shift_preds    = []
        all_shift_labels   = []
        all_shift_probs    = []
        all_has_prev       = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
                tensor_batch = self._to_device(batch)

                if hasattr(self.model, "reset_memory"):
                    self.model.reset_memory()

                outputs = self.model(**tensor_batch)

                emotion_preds = torch.argmax(outputs["emotion_logits"], dim=1)
                shift_probs   = torch.sigmoid(outputs["shift_logits"]).cpu().numpy()

                all_emotion_preds.extend(emotion_preds.cpu().numpy())
                all_emotion_labels.extend(batch["emotion_labels"].cpu().numpy())
                all_shift_probs.extend(shift_probs)
                all_shift_preds.extend((shift_probs > 0.5).astype(int))
                all_shift_labels.extend(batch["shift_labels"].cpu().numpy())
                all_has_prev.extend(batch["has_prev_mask"].cpu().numpy())

        ep = np.array(all_emotion_preds)
        el = np.array(all_emotion_labels)
        sp = np.array(all_shift_preds)
        sl = np.array(all_shift_labels)
        sr = np.array(all_shift_probs)
        hp = np.array(all_has_prev)

        valid_mask  = (hp == 1) & (sl != -100)
        vsp, vsl, vsr = sp[valid_mask], sl[valid_mask], sr[valid_mask]

        emotion_f1  = f1_score(el, ep, average="macro")
        shift_f1    = f1_score(vsl, vsp) if len(vsl) > 0 else 0.0
        shift_auprc = 0.0
        if len(vsl) > 0:
            prec, rec, _ = precision_recall_curve(vsl, vsr)
            shift_auprc  = auc(rec, prec)

        return {
            "emotion_macro_f1": emotion_f1,
            "shift_f1":         shift_f1,
            "shift_auprc":      shift_auprc,
            "emotion_preds":    ep,
            "emotion_labels":   el,
            "shift_preds":      sp,
            "shift_labels":     sl,
            "shift_probs":      sr,
            "valid_shift_mask": valid_mask,
        }

    # ── Full Experiment ──

    def run_experiment(self, num_epochs: int = 3, model_name: str = "model") -> dict:
        """Train for num_epochs, evaluate on val after each epoch, test on best."""
        best_val_f1 = 0.0
        results = {"train_losses": [], "val_metrics": [], "test_metrics": None}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss  = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, "Validation")

            results["train_losses"].append(train_loss)
            results["val_metrics"].append(val_metrics)

            print(f"  Train Loss  : {train_loss:.4f}")
            print(f"  Val Emo F1  : {val_metrics['emotion_macro_f1']:.4f}")
            print(f"  Val Shift F1: {val_metrics['shift_f1']:.4f}")
            print(f"  Val AUPRC   : {val_metrics['shift_auprc']:.4f}")

            combined = val_metrics["emotion_macro_f1"] + val_metrics["shift_f1"]
            if combined > best_val_f1:
                best_val_f1 = combined
                torch.save(self.model.state_dict(), f"{model_name}_best.pth")

        self.model.load_state_dict(torch.load(f"{model_name}_best.pth"))
        test_metrics = self.evaluate(self.test_loader, "Test")
        results["test_metrics"] = test_metrics

        print(f"\nFinal Test Results for {model_name}:")
        print(f"  Emotion F1 : {test_metrics['emotion_macro_f1']:.4f}")
        print(f"  Shift F1   : {test_metrics['shift_f1']:.4f}")
        print(f"  AUPRC      : {test_metrics['shift_auprc']:.4f}")

        return results

    # ── Helpers ──

    def _to_device(self, batch: dict) -> dict:
        """Move tensor values to device; leave lists and strings as-is."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


# ─── Enhanced Runner (adds per-emotion, calibration, breakdown) ───────────────

class EnhancedExperimentRunner(ExperimentRunner):
    """
    Extends ExperimentRunner with:
    - Per-emotion F1 scores
    - Shift calibration data
    - Confusion matrix
    - Breakdown analyses by dialogue length, turn position, speaker
    """

    def enhanced_evaluate(self, loader, split_name: str = "Validation") -> dict:
        """
        Full evaluation with detailed diagnostic outputs.

        Adds to the base evaluate() dict:
            per_emotion_f1, emotion_confusion_matrix,
            shift_calibration, breakdowns (dict)
        """
        self.model.eval()

        all_emotion_preds  = []
        all_emotion_labels = []
        all_shift_preds    = []
        all_shift_labels   = []
        all_shift_probs    = []
        all_has_prev       = []
        all_speakers       = []
        all_dialog_ids     = []
        all_turn_indices   = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Enhanced Eval {split_name}"):
                tensor_batch = self._to_device(batch)

                if hasattr(self.model, "reset_memory"):
                    self.model.reset_memory()

                outputs     = self.model(**tensor_batch)
                emo_preds   = torch.argmax(outputs["emotion_logits"], dim=1)
                shift_probs = torch.sigmoid(outputs["shift_logits"]).cpu().numpy()

                all_emotion_preds.extend(emo_preds.cpu().numpy())
                all_emotion_labels.extend(batch["emotion_labels"].cpu().numpy())
                all_shift_probs.extend(shift_probs)
                all_shift_preds.extend((shift_probs > 0.5).astype(int))
                all_shift_labels.extend(batch["shift_labels"].cpu().numpy())
                all_has_prev.extend(batch["has_prev_mask"].cpu().numpy())

                # Optional diagnostic fields
                all_speakers.extend(batch.get("speaker_ids", []) or [])
                all_dialog_ids.extend(batch.get("dialog_ids", []) or [])
                all_turn_indices.extend(batch.get("turn_indices", []) or [])

        ep = np.array(all_emotion_preds)
        el = np.array(all_emotion_labels)
        sp = np.array(all_shift_preds)
        sl = np.array(all_shift_labels)
        sr = np.array(all_shift_probs)
        hp = np.array(all_has_prev)

        valid_mask = (hp == 1) & (sl != -100)
        vsp, vsl, vsr = sp[valid_mask], sl[valid_mask], sr[valid_mask]

        # Core metrics
        emotion_f1  = f1_score(el, ep, average="macro")
        per_emo_f1  = f1_score(el, ep, average=None, labels=list(range(7)))
        conf_matrix = confusion_matrix(el, ep, labels=list(range(7)))

        shift_f1    = f1_score(vsl, vsp) if len(vsl) > 0 else 0.0
        shift_auprc = 0.0
        calibration = None
        if len(vsl) > 0 and len(np.unique(vsl)) > 1:
            prec, rec, _ = precision_recall_curve(vsl, vsr)
            shift_auprc  = auc(rec, prec)
            try:
                prob_true, prob_pred = calibration_curve(vsl, vsr, n_bins=10)
                calibration = (prob_true, prob_pred)
            except Exception:
                pass

        # Breakdown analyses
        breakdowns = self._compute_breakdowns(
            ep, el, all_speakers, all_dialog_ids, all_turn_indices
        )

        return {
            "emotion_macro_f1":        emotion_f1,
            "per_emotion_f1":          per_emo_f1,
            "emotion_confusion_matrix": conf_matrix,
            "shift_f1":                shift_f1,
            "shift_auprc":             shift_auprc,
            "shift_calibration":       calibration,
            "emotion_preds":           ep,
            "emotion_labels":          el,
            "shift_preds":             sp,
            "shift_labels":            sl,
            "shift_probs":             sr,
            "valid_shift_mask":        valid_mask,
            "breakdowns":              breakdowns,
        }

    def _compute_breakdowns(self, emotion_preds, emotion_labels,
                            speakers, dialog_ids, turn_indices) -> dict:
        """Compute breakdown analyses by dialogue length, turn position, speaker."""
        import pandas as pd
        from collections import defaultdict

        breakdowns = {}

        # ── Dialogue-length breakdown ──
        dialog_turns = defaultdict(list)
        for i, d_id in enumerate(dialog_ids):
            dialog_turns[d_id].append(i)

        length_rows = []
        for d_id, indices in dialog_turns.items():
            if not indices:
                continue
            mask = np.array(indices)
            f1   = f1_score(emotion_labels[mask], emotion_preds[mask],
                            average="macro", zero_division=0)
            length_rows.append({
                "dialog_id":  d_id,
                "length":     len(indices),
                "emotion_f1": f1,
                "category":   "short" if len(indices) <= 3 else "long",
            })
        breakdowns["dialogue_length"] = pd.DataFrame(length_rows)

        # ── Turn-position breakdown ──
        if turn_indices:
            max_turn = max(turn_indices)
            turn_rows = []
            for i, t_idx in enumerate(turn_indices):
                pos = "early" if (max_turn == 0 or t_idx < max_turn // 2) else "late"
                turn_rows.append({
                    "turn_idx":     t_idx,
                    "position":     pos,
                    "emotion_pred": emotion_preds[i],
                    "emotion_label": emotion_labels[i],
                })
            breakdowns["turn_position"] = pd.DataFrame(turn_rows)

        # ── Speaker breakdown ──
        if speakers:
            speaker_rows = []
            for spk in set(speakers):
                mask = np.array([s == spk for s in speakers])
                if mask.sum() > 0:
                    speaker_rows.append({
                        "speaker":    spk,
                        "emotion_f1": f1_score(emotion_labels[mask], emotion_preds[mask],
                                               average="macro", zero_division=0),
                        "num_turns":  int(mask.sum()),
                    })
            breakdowns["speaker"] = pd.DataFrame(speaker_rows)

        return breakdowns

    def run_enhanced_experiment(self, num_epochs: int = 3,
                                model_name: str = "model") -> dict:
        """Train and evaluate with full diagnostics."""
        best_val_f1 = 0.0
        results = {"train_losses": [], "val_metrics": [], "test_metrics": None}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss  = self.train_epoch()
            val_metrics = self.enhanced_evaluate(self.val_loader, "Validation")

            results["train_losses"].append(train_loss)
            results["val_metrics"].append(val_metrics)

            print(f"  Train Loss  : {train_loss:.4f}")
            print(f"  Val Emo F1  : {val_metrics['emotion_macro_f1']:.4f}")
            print(f"  Val Shift F1: {val_metrics['shift_f1']:.4f}")
            print(f"  Val AUPRC   : {val_metrics['shift_auprc']:.4f}")

            combined = val_metrics["emotion_macro_f1"] + val_metrics["shift_f1"]
            if combined > best_val_f1:
                best_val_f1 = combined
                torch.save(self.model.state_dict(), f"{model_name}_best.pth")

        self.model.load_state_dict(torch.load(f"{model_name}_best.pth"))
        test_metrics = self.enhanced_evaluate(self.test_loader, "Test")
        results["test_metrics"] = test_metrics

        print(f"\nFinal Test Results — {model_name}")
        self._print_results(test_metrics)
        return results

    @staticmethod
    def _print_results(metrics: dict) -> None:
        emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        print(f"  Emotion Macro-F1 : {metrics['emotion_macro_f1']:.4f}")
        print(f"  Shift F1         : {metrics['shift_f1']:.4f}")
        print(f"  Shift AUPRC      : {metrics['shift_auprc']:.4f}")
        print("  Per-Emotion F1:")
        for i, emo in enumerate(emotions):
            print(f"    {emo:<10}: {metrics['per_emotion_f1'][i]:.4f}")


# ─── B0 Evaluation Helper ─────────────────────────────────────────────────────

def evaluate_b0_baseline(baseline, test_loader) -> dict:
    """
    Evaluate a B0MajorityBaseline object on a DataLoader.
    Returns the same dict structure as ExperimentRunner.evaluate().
    """
    all_ep, all_el = [], []
    all_sp, all_sl = [], []
    all_hp = []

    for batch in tqdm(test_loader, desc="Evaluating B0"):
        outputs = baseline.predict(batch)
        all_ep.extend(outputs["emotion_preds"].numpy())
        all_el.extend(batch["emotion_labels"].numpy())
        all_sp.extend(outputs["shift_preds"].numpy())
        all_sl.extend(batch["shift_labels"].numpy())
        all_hp.extend(batch["has_prev_mask"].numpy())

    ep = np.array(all_ep)
    el = np.array(all_el)
    sp = np.array(all_sp)
    sl = np.array(all_sl)
    hp = np.array(all_hp)

    valid_mask = (hp == 1) & (sl != -100)
    vsp, vsl   = sp[valid_mask], sl[valid_mask]

    emotion_f1 = f1_score(el, ep, average="macro")
    shift_f1   = f1_score(vsl, vsp) if len(vsl) > 0 else 0.0

    return {
        "emotion_macro_f1": emotion_f1,
        "shift_f1":         shift_f1,
        "shift_auprc":      0.0,       # no probabilities for majority baseline
        "emotion_preds":    ep,
        "emotion_labels":   el,
        "shift_preds":      sp,
        "shift_labels":     sl,
        "shift_probs":      np.zeros_like(sp, dtype=float),
        "valid_shift_mask": valid_mask,
    }

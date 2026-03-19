"""
analyze.py
----------
ExperimentAnalyzer and EnhancedExperimentAnalyzer:
aggregate results, build ablation tables, and produce
the comprehensive diagnostic plots from the paper (Figure 1).

Paper: "Speaker-State Memory for Emotion-Shift Modeling in Dyadic Dialogues"
Author: Tulika Sharma (stulika029@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import f1_score


EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# ─── Base Analyzer ────────────────────────────────────────────────────────────

class ExperimentAnalyzer:
    """
    Collects results from multiple experiment runs and produces
    comparison tables and learning-curve plots.
    """

    def __init__(self):
        self.results: dict = {}

    def add_experiment(self, name: str, results: dict) -> None:
        self.results[name] = results

    def create_ablation_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising Emotion F1, Shift F1, AUPRC for all models."""
        rows = []
        for model_name, result in self.results.items():
            m = result["test_metrics"]
            rows.append({
                "Model":            model_name,
                "Emotion Macro-F1": f"{m['emotion_macro_f1']:.4f}",
                "Shift F1":         f"{m['shift_f1']:.4f}",
                "Shift AUPRC":      f"{m['shift_auprc']:.4f}",
            })
        return pd.DataFrame(rows)

    def plot_learning_curves(self) -> None:
        """Plot training loss and validation Emotion F1 across epochs for all models."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for name, result in self.results.items():
            axes[0].plot(result["train_losses"], label=name)
            val_f1 = [m["emotion_macro_f1"] for m in result["val_metrics"]]
            axes[1].plot(val_f1, label=name)

        for ax, title, ylabel in zip(
            axes,
            ["Training Loss", "Validation Emotion F1"],
            ["Loss", "Macro F1"],
        ):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.show()

    def breakdown_analysis(self, dataset, model_name: str) -> pd.DataFrame:
        """Analyse performance by dialogue length for a given model."""
        if model_name not in self.results:
            return pd.DataFrame()

        preds  = self.results[model_name]["test_metrics"]["emotion_preds"]
        labels = self.results[model_name]["test_metrics"]["emotion_labels"]

        dialog_lengths = defaultdict(list)
        for i, ex in enumerate(dataset.examples):
            dialog_lengths[ex["dialog_id"]].append(i)

        rows = []
        for d_id, indices in dialog_lengths.items():
            if not indices:
                continue
            ep = preds[np.array(indices)]
            el = labels[np.array(indices)]
            f1 = f1_score(el, ep, average="macro", zero_division=0)
            rows.append({
                "dialog_id":  d_id,
                "length":     len(indices),
                "emotion_f1": f1,
                "category":   "short" if len(indices) <= 3 else "long",
            })

        df = pd.DataFrame(rows)
        print(f"\nPerformance by Dialogue Length ({model_name}):")
        print(df.groupby("category")["emotion_f1"].mean())
        return df


# ─── Enhanced Analyzer ────────────────────────────────────────────────────────

class EnhancedExperimentAnalyzer(ExperimentAnalyzer):
    """
    Extends ExperimentAnalyzer with:
    - Per-emotion F1 in the ablation table
    - Comprehensive 6-panel diagnostic plots (paper Figure 1)
    """

    def create_comprehensive_ablation_table(self) -> pd.DataFrame:
        """Ablation table including per-emotion F1 columns."""
        rows = []
        for name, result in self.results.items():
            m   = result["test_metrics"]
            row = {
                "Model":            name,
                "Emotion Macro-F1": f"{m['emotion_macro_f1']:.4f}",
                "Shift F1":         f"{m['shift_f1']:.4f}",
                "Shift AUPRC":      f"{m['shift_auprc']:.4f}",
            }
            if "per_emotion_f1" in m:
                for i, emo in enumerate(EMOTION_LABELS):
                    row[f"{emo}_F1"] = f"{m['per_emotion_f1'][i]:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Figure 1: Comprehensive Diagnostic Plot ──

    def plot_comprehensive_analysis(self, model_name: str) -> None:
        """
        Reproduce Figure 1 from the paper: a 2×3 grid of diagnostic plots
        for the specified model.

        Subplots:
          (A) Confusion Matrix
          (B) Shift Calibration Curve
          (C) Emotion F1 by Dialogue Length
          (D) Per-Emotion F1 Bar Chart
          (E) Emotion F1 by Turn Position
          (F) Emotion F1 by Speaker
        """
        if model_name not in self.results:
            print(f"Model '{model_name}' not found.")
            return

        m   = self.results[model_name]["test_metrics"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Comprehensive Analysis: {model_name}", fontsize=16)

        self._plot_confusion_matrix(m.get("emotion_confusion_matrix"), axes[0, 0])
        self._plot_calibration_curve(m.get("shift_calibration"), axes[0, 1])
        self._plot_dialogue_length(m.get("breakdowns", {}).get("dialogue_length"), axes[0, 2])
        self._plot_per_emotion_f1(m.get("per_emotion_f1"), axes[1, 0])
        self._plot_turn_position(m.get("breakdowns", {}).get("turn_position"), axes[1, 1])
        self._plot_speaker(m.get("breakdowns", {}).get("speaker"), axes[1, 2])

        plt.tight_layout()
        plt.show()

    # ── Individual Plot Helpers ──

    @staticmethod
    def _plot_confusion_matrix(cm, ax) -> None:
        if cm is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title("Confusion Matrix")
            return
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    @staticmethod
    def _plot_calibration_curve(calibration, ax) -> None:
        ax.set_title("Shift Calibration Curve")
        if calibration is None:
            ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
            return
        prob_true, prob_pred = calibration
        ax.plot(prob_pred, prob_true, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_dialogue_length(length_df, ax) -> None:
        ax.set_title("Performance by Dialogue Length")
        if length_df is None or length_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        sns.boxplot(data=length_df, x="category", y="emotion_f1", ax=ax)
        ax.set_xlabel("Dialogue Length Category")
        ax.set_ylabel("Emotion F1 Score")

    @staticmethod
    def _plot_per_emotion_f1(per_emotion_f1, ax) -> None:
        ax.set_title("Per-Emotion F1 Scores")
        if per_emotion_f1 is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        colors = plt.cm.Set3(np.linspace(0, 1, len(EMOTION_LABELS)))
        bars = ax.bar(EMOTION_LABELS, per_emotion_f1, color=colors)
        ax.set_ylabel("F1 Score")
        ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    @staticmethod
    def _plot_turn_position(turn_df, ax) -> None:
        ax.set_title("Performance by Turn Position")
        if turn_df is None or turn_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        pos_f1 = turn_df.groupby("position").apply(
            lambda x: f1_score(x["emotion_label"], x["emotion_pred"],
                               average="macro", zero_division=0)
        )
        pos_f1.plot(kind="bar", ax=ax, color=["skyblue", "lightcoral"])
        ax.set_ylabel("Emotion F1 Score")
        ax.set_xlabel("Turn Position")

    @staticmethod
    def _plot_speaker(speaker_df, ax) -> None:
        ax.set_title("Performance by Speaker")
        if speaker_df is None or speaker_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        ax.bar(speaker_df["speaker"], speaker_df["emotion_f1"],
               color=["lightblue", "lightpink"])
        ax.set_ylabel("Emotion F1 Score")
        ax.set_xlabel("Speaker")

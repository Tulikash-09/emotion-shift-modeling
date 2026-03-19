"""
utils.py
--------
Utilities for saving/loading experiments, running model grids,
and helper functions for the emotion-shift modeling codebase.

Paper: "Speaker-State Memory for Emotion-Shift Modeling in Dyadic Dialogues"
Author: Tulika Sharma (stulika029@gmail.com)
"""

import os
import pickle
from datetime import datetime

import torch


# ─── Experiment Saver ─────────────────────────────────────────────────────────

class ExperimentSaver:
    """
    Saves and loads experiment results, model weights, and result tables.

    Designed for both local use and Google Colab (optionally mounts Drive).

    Parameters
    ----------
    base_dir : str
        Root directory for all experiment artefacts.
    use_drive : bool
        If True (Colab), mounts Google Drive and saves under MyDrive/base_dir.
    """

    def __init__(self, base_dir: str = "emotion_experiments", use_drive: bool = False):
        if use_drive:
            try:
                from google.colab import drive
                drive.mount("/content/drive")
                self.base_path = f"/content/drive/MyDrive/{base_dir}"
            except ImportError:
                print("google.colab not available — saving locally instead.")
                self.base_path = base_dir
        else:
            self.base_path = base_dir

        self.results_dir = os.path.join(self.base_path, "results")
        self.models_dir  = os.path.join(self.base_path, "models")
        self.tables_dir  = os.path.join(self.base_path, "tables")

        for d in [self.results_dir, self.models_dir, self.tables_dir]:
            os.makedirs(d, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"ExperimentSaver initialised at: {self.base_path}")

    # ── Save ──

    def save_baseline_experiments(self, analyzer, b0_baseline=None) -> str:
        print("Saving baseline experiments…")
        path = os.path.join(self.results_dir, f"baseline_analyzer_{self.timestamp}.pkl")
        with open(path, "wb") as f:
            pickle.dump(analyzer, f)

        if b0_baseline is not None:
            b0_path = os.path.join(self.models_dir, f"b0_baseline_{self.timestamp}.pkl")
            with open(b0_path, "wb") as f:
                pickle.dump(b0_baseline, f)

        table = analyzer.create_ablation_table()
        table.to_csv(os.path.join(self.tables_dir, f"baseline_results_{self.timestamp}.csv"), index=False)
        print(f"  Saved to {path}")
        return path

    def save_memory_experiments(self, analyzer, model_instances: dict = None):
        print("Saving memory experiments…")
        path = os.path.join(self.results_dir, f"memory_analyzer_{self.timestamp}.pkl")
        with open(path, "wb") as f:
            pickle.dump(analyzer, f)

        saved_models = {}
        if model_instances:
            for name, model in model_instances.items():
                mp = os.path.join(self.models_dir, f"{name}_weights_{self.timestamp}.pth")
                torch.save(model.state_dict(), mp)
                saved_models[name] = mp
                print(f"  Saved {name} → {mp}")

        if hasattr(analyzer, "create_comprehensive_ablation_table"):
            table = analyzer.create_comprehensive_ablation_table()
        else:
            table = analyzer.create_ablation_table()
        table.to_csv(os.path.join(self.tables_dir, f"memory_results_{self.timestamp}.csv"), index=False)
        print(f"  Saved analyzer → {path}")
        return path, saved_models

    def save_checkpoint(self, model, optimizer, epoch: int, loss: float, name: str) -> str:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        path = os.path.join(self.models_dir, f"{name}_checkpoint_epoch{epoch}.pth")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        return path

    # ── Load ──

    def load_baseline_experiments(self, timestamp: str = None):
        if timestamp is None:
            files = sorted(f for f in os.listdir(self.results_dir) if f.startswith("baseline_analyzer"))
            if not files:
                raise FileNotFoundError("No baseline experiments found.")
            timestamp = files[-1].split("_")[-1].replace(".pkl", "")

        path = os.path.join(self.results_dir, f"baseline_analyzer_{timestamp}.pkl")
        with open(path, "rb") as f:
            analyzer = pickle.load(f)

        b0_path = os.path.join(self.models_dir, f"b0_baseline_{timestamp}.pkl")
        b0 = None
        if os.path.exists(b0_path):
            with open(b0_path, "rb") as f:
                b0 = pickle.load(f)

        print(f"Loaded baseline experiments (timestamp: {timestamp})")
        return analyzer, b0

    def load_memory_experiments(self, timestamp: str = None):
        if timestamp is None:
            files = sorted(f for f in os.listdir(self.results_dir) if f.startswith("memory_analyzer"))
            if not files:
                raise FileNotFoundError("No memory experiments found.")
            timestamp = files[-1].split("_")[-1].replace(".pkl", "")

        path = os.path.join(self.results_dir, f"memory_analyzer_{timestamp}.pkl")
        with open(path, "rb") as f:
            analyzer = pickle.load(f)

        print(f"Loaded memory experiments (timestamp: {timestamp})")
        return analyzer

    def list_saved_experiments(self) -> None:
        baseline = sorted(f for f in os.listdir(self.results_dir) if f.startswith("baseline"))
        memory   = sorted(f for f in os.listdir(self.results_dir) if f.startswith("memory"))
        print("Saved Experiments:")
        print(f"  Baseline ({len(baseline)}): " + (", ".join(baseline) or "none"))
        print(f"  Memory   ({len(memory)}): "   + (", ".join(memory)   or "none"))


# ─── Experiment Grid Runner ───────────────────────────────────────────────────

def run_baseline_experiment_grid(
    train_loader_b1, val_loader_b1, test_loader_b1,
    train_loader_b2, val_loader_b2, test_loader_b2,
    device, roberta_tokenizer, saver=None,
):
    """
    Run B0, B1, and B2 baseline experiments and return an ExperimentAnalyzer.

    Parameters
    ----------
    train/val/test_loader_b1 : DataLoader  — utterance-only collate
    train/val/test_loader_b2 : DataLoader  — contextual collate
    device : torch.device
    roberta_tokenizer : RobertaTokenizer with special tokens added
    saver : ExperimentSaver or None
    """
    from src.models import B0MajorityBaseline, B1UtteranceOnlyModel, B2ContextualModel
    from src.train  import ExperimentRunner, evaluate_b0_baseline
    from src.analyze import ExperimentAnalyzer

    analyzer = ExperimentAnalyzer()

    # B0
    print("Running B0 Majority Baseline…")
    b0 = B0MajorityBaseline()
    b0.fit(train_loader_b1.dataset)
    b0_metrics = evaluate_b0_baseline(b0, test_loader_b1)
    analyzer.add_experiment("B0_Majority", {
        "train_losses": [0], "val_metrics": [b0_metrics], "test_metrics": b0_metrics
    })

    configs = [
        {"name": "B1_UtteranceOnly_λ0.5", "cls": B1UtteranceOnlyModel,
         "kwargs": {"lambda_shift": 0.5}, "loaders": (train_loader_b1, val_loader_b1, test_loader_b1)},
        {"name": "B2_Contextual_λ0.5",   "cls": B2ContextualModel,
         "kwargs": {"lambda_shift": 0.5}, "loaders": (train_loader_b2, val_loader_b2, test_loader_b2)},
    ]

    trained = {}
    for cfg in configs:
        print(f"\n{'='*50}\nRunning: {cfg['name']}\n{'='*50}")
        model = cfg["cls"](**cfg["kwargs"])
        model.roberta.resize_token_embeddings(len(roberta_tokenizer))
        model = model.to(device)

        tr, va, te = cfg["loaders"]
        runner  = ExperimentRunner(model, tr, va, te, device)
        results = runner.run_experiment(num_epochs=3, model_name=cfg["name"])
        analyzer.add_experiment(cfg["name"], results)
        trained[cfg["name"]] = model

        if saver:
            saver.save_checkpoint(model, runner.optimizer, 3,
                                  results["train_losses"][-1], cfg["name"])

    if saver:
        saver.save_baseline_experiments(analyzer, b0)

    return analyzer, trained


def run_memory_experiment_grid(
    train_loader_b3, val_loader_b3, test_loader_b3,
    device, roberta_tokenizer, saver=None,
):
    """
    Run all B3 memory model variants and return an EnhancedExperimentAnalyzer.
    """
    from src.models  import MemoryAugmentedModel, GRUMemoryWithSentimentModel
    from src.train   import EnhancedExperimentRunner
    from src.analyze import EnhancedExperimentAnalyzer

    analyzer = EnhancedExperimentAnalyzer()
    trained  = {}

    configs = [
        {"name": "B3_Memory_GRU_λ0.5",
         "cls": MemoryAugmentedModel,
         "kwargs": {"lambda_shift": 0.5, "memory_type": "GRU", "memory_dim": 128}},
        {"name": "B3_Memory_Gated_λ0.5",
         "cls": MemoryAugmentedModel,
         "kwargs": {"lambda_shift": 0.5, "memory_type": "Gated", "memory_dim": 128}},
        {"name": "B3_Memory_GRU_Trunc5_λ0.5",
         "cls": MemoryAugmentedModel,
         "kwargs": {"lambda_shift": 0.5, "memory_type": "GRU", "memory_dim": 128, "truncate_memory": 5}},
        {"name": "B3_Memory_GRU_Sentiment_λ0.5",
         "cls": GRUMemoryWithSentimentModel,
         "kwargs": {"lambda_shift": 0.5, "memory_dim": 128}},
    ]

    for cfg in configs:
        print(f"\n{'='*50}\nRunning: {cfg['name']}\n{'='*50}")
        model = cfg["cls"](**cfg["kwargs"])
        model.roberta.resize_token_embeddings(len(roberta_tokenizer))

        runner  = EnhancedExperimentRunner(model, train_loader_b3, val_loader_b3,
                                           test_loader_b3, device)
        results = runner.run_enhanced_experiment(num_epochs=3, model_name=cfg["name"])
        analyzer.add_experiment(cfg["name"], results)
        trained[cfg["name"]] = model
        analyzer.plot_comprehensive_analysis(cfg["name"])

    if saver:
        saver.save_memory_experiments(analyzer, trained)

    return analyzer, trained

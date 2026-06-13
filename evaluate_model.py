"""
evaluate_model.py
-----------------
P3 Responsibility: Model accuracy evaluation.

Compares predicted sentiment labels against the ground truth (true_sentiment)
column generated alongside sample data.

Metrics:
  - Accuracy, Precision, Recall, F1 per class
  - Confusion matrix
  - VADER vs BERT vs Ensemble comparison
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── Always run relative to this script folder ────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))


INPUT_CSV   = "output/sentiment_results.csv"
OUTPUT_JSON = "output/evaluation_report.json"


def evaluate(df: pd.DataFrame, pred_col: str, true_col: str = "true_sentiment") -> dict:
    """Compute classification metrics for a given prediction column."""
    valid = df[df[true_col] != "unknown"].copy()
    if len(valid) == 0:
        return {"error": "No ground truth labels found."}

    y_true = valid[true_col]
    y_pred = valid[pred_col]

    acc    = round(accuracy_score(y_true, y_pred) * 100, 2)
    report = classification_report(
        y_true, y_pred,
        labels=["positive", "negative", "neutral"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true, y_pred,
        labels=["positive", "negative", "neutral"]
    ).tolist()

    return {
        "accuracy":   acc,
        "f1_macro":   round(report["macro avg"]["f1-score"] * 100, 2),
        "per_class":  {
            cls: {
                "precision": round(report[cls]["precision"] * 100, 2),
                "recall":    round(report[cls]["recall"] * 100, 2),
                "f1":        round(report[cls]["f1-score"] * 100, 2),
                "support":   int(report[cls]["support"]),
            }
            for cls in ["positive", "negative", "neutral"]
            if cls in report
        },
        "confusion_matrix": {
            "labels": ["positive", "negative", "neutral"],
            "matrix": cm,
        },
        "n_evaluated": len(valid),
    }


def run_evaluation() -> dict:
    os.makedirs("output", exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    print(f"Evaluating {len(df)} reviews...")

    report = {
        "vader":    evaluate(df, "vader_label"),
        "bert":     evaluate(df, "bert_label") if "bert_label" in df.columns else None,
        "ensemble": evaluate(df, "final_label"),
    }

    # Print summary table
    print("\n── Accuracy Summary ──────────────────────────────────────────────")
    for model, metrics in report.items():
        if metrics is None:
            print(f"  {model:10s} → not available")
            continue
        if "error" in metrics:
            print(f"  {model:10s} → {metrics['error']}")
            continue
        print(f"  {model:10s} → Accuracy: {metrics['accuracy']:5.1f}%  "
              f"F1 macro: {metrics['f1_macro']:5.1f}%  "
              f"(n={metrics['n_evaluated']})")

    # Print per-class breakdown for ensemble
    ens = report["ensemble"]
    if ens and "per_class" in ens:
        print("\n── Ensemble Per-Class ────────────────────────────────────────────")
        for cls, m in ens["per_class"].items():
            print(f"  {cls:10s} → P:{m['precision']:5.1f}%  "
                  f"R:{m['recall']:5.1f}%  F1:{m['f1']:5.1f}%  "
                  f"(support={m['support']})")

    # Confusion matrix
    if ens and "confusion_matrix" in ens:
        cm = ens["confusion_matrix"]
        print("\n── Confusion Matrix (Ensemble) ───────────────────────────────────")
        print("             " + "  ".join(f"{l[:4]:>6}" for l in cm["labels"]))
        for i, row in enumerate(cm["matrix"]):
            print(f"  {cm['labels'][i]:10s} " +
                  "  ".join(f"{v:6d}" for v in row))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅  Evaluation report → {OUTPUT_JSON}")
    return report


if __name__ == "__main__":
    run_evaluation()

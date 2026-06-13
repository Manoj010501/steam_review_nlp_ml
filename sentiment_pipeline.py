"""
sentiment_pipeline.py
---------------------
P3 Responsibility: Sentiment scoring pipeline using VADER + BERT.

- VADER  → fast rule-based scorer (good for short social text)
- BERT   → transformer classifier via HuggingFace (more accurate)
- Final  → ensemble of both scores

Outputs a CSV with sentiment label + 0-100 confidence score for each review.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import json
import os

# ── Always run relative to this script folder ────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/sample_reviews.csv"
OUTPUT_CSV = "output/sentiment_results.csv"
OUTPUT_JSON = "output/sentiment_results.json"

# Use a lightweight sentiment model (works offline after first download)
BERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


# ── VADER Scorer ───────────────────────────────────────────────────────────────

class VADERScorer:
    """Wraps VADER to return a label + 0-100 compound score."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]         # -1 to +1

        # Map compound to 0-100
        normalized = round((compound + 1) / 2 * 100, 2)

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "vader_label":    label,
            "vader_score":    normalized,
            "vader_compound": round(compound, 4),
            "vader_pos":      round(scores["pos"], 4),
            "vader_neg":      round(scores["neg"], 4),
            "vader_neu":      round(scores["neu"], 4),
        }


# ── BERT Scorer ────────────────────────────────────────────────────────────────

class BERTScorer:
    """
    Uses HuggingFace distilbert SST-2 fine-tuned model.
    First run will download ~250MB model weights.
    """

    def __init__(self):
        print("Loading BERT model (downloads on first run)...")
        self.classifier = pipeline(
            "sentiment-analysis",
            model=BERT_MODEL,
            truncation=True,
            max_length=512,
        )
        print("✅  BERT model loaded.")

    def score(self, text: str) -> dict:
        result = self.classifier(text)[0]
        raw_label = result["label"]          # "POSITIVE" or "NEGATIVE"
        confidence = round(result["score"] * 100, 2)

        # Map to our 3-class + normalise score to 0-100
        if raw_label == "POSITIVE":
            label = "positive"
            score = confidence                      # already 50-100
        else:
            label = "negative"
            score = round(100 - confidence, 2)      # invert so 0 = very negative

        return {
            "bert_label": label,
            "bert_score": score,
        }


# ── Ensemble ───────────────────────────────────────────────────────────────────

def ensemble_label(vader_label: str, bert_label: str,
                   vader_score: float, bert_score: float) -> dict:
    """
    Combine VADER + BERT.
    - If both agree → use that label.
    - If they disagree → trust BERT (higher weight).
    - Final score = weighted average (VADER 35%, BERT 65%).
    """
    final_score = round(vader_score * 0.35 + bert_score * 0.65, 2)

    if vader_label == bert_label:
        final_label = vader_label
    else:
        # Disagreement: trust BERT but mark as lower confidence
        final_label = bert_label

    # Classify from final_score for 3-class
    if final_score >= 60:
        final_label = "positive"
    elif final_score <= 40:
        final_label = "negative"
    else:
        final_label = "neutral"

    return {
        "final_label": final_label,
        "final_score": final_score,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_sentiment_pipeline(use_bert: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load CSV → score each review → save results.

    Args:
        use_bert: Set False to skip BERT (faster, VADER only).
    """
    os.makedirs("output", exist_ok=True)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} reviews from {INPUT_CSV}")

    # Init scorers
    vader = VADERScorer()
    bert  = BERTScorer() if use_bert else None

    results = []

    for i, row in df.iterrows():
        text = str(row["review_text"])

        v = vader.score(text)

        if bert:
            b = bert.score(text)
            e = ensemble_label(
                v["vader_label"], b["bert_label"],
                v["vader_score"], b["bert_score"]
            )
        else:
            # VADER only fallback
            b = {"bert_label": v["vader_label"], "bert_score": v["vader_score"]}
            e = {"final_label": v["vader_label"], "final_score": v["vader_score"]}

        record = {
            "review_id":      row["review_id"],
            "app_id":         row["app_id"],
            "game_name":      row["game_name"],
            "review_text":    text,
            "month":          row["month"],
            "playtime_hours": row["playtime_hours"],
            "true_sentiment": row.get("true_sentiment", "unknown"),
            **v,
            **b,
            **e,
        }
        results.append(record)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)} reviews...")

    result_df = pd.DataFrame(results)

    # Save
    result_df.to_csv(OUTPUT_CSV, index=False)
    result_df.to_json(OUTPUT_JSON, orient="records", indent=2)

    print(f"\n✅  Sentiment results saved → {OUTPUT_CSV}")
    print(result_df["final_label"].value_counts())
    return result_df


if __name__ == "__main__":
    # Set use_bert=False to skip downloading the transformer model during testing
    df = run_sentiment_pipeline(use_bert=False)
    print("\nSample output:")
    print(df[["review_id", "game_name", "vader_label", "final_label",
              "final_score", "true_sentiment"]].head(10))
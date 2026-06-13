"""
topic_modeling.py
-----------------
P3 Responsibility: Topic cluster extraction using BERTopic.

BERTopic pipeline:
  1. Embed reviews with sentence-transformers
  2. Reduce dims with UMAP
  3. Cluster with HDBSCAN
  4. Extract topic keywords with c-TF-IDF

Outputs topic labels, topic sentiment, and per-review topic assignment.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# ── Always run relative to this script folder ────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_CSV   = "output/sentiment_results.csv"   # output from sentiment_pipeline
OUTPUT_CSV  = "output/topics_results.csv"
OUTPUT_JSON = "output/topics_results.json"
TOPICS_JSON = "output/topic_summary.json"

# Pre-defined topic category keywords for mapping raw BERTopic labels → themes
THEME_KEYWORDS = {
    "gameplay":     ["combat", "controls", "mechanic", "gameplay", "fight",
                     "skill", "build", "weapon", "system", "fun", "boring",
                     "repetitive", "satisfying", "clunky"],
    "performance":  ["fps", "crash", "stutter", "optimization", "lag", "bug",
                     "freeze", "performance", "settings", "hardware", "ram",
                     "gpu", "cpu", "runs", "stable"],
    "story":        ["story", "narrative", "character", "plot", "ending",
                     "voice", "acting", "cutscene", "lore", "world",
                     "immersive", "writing", "dialogue"],
    "value":        ["price", "worth", "expensive", "cheap", "sale",
                     "content", "hours", "dlc", "microtransaction",
                     "pay", "money", "refund"],
    "developer":    ["developer", "dev", "update", "patch", "support",
                     "abandoned", "community", "roadmap", "fix", "listen"],
    "multiplayer":  ["multiplayer", "online", "server", "matchmaking",
                     "co-op", "pvp", "friends", "lobby", "ping", "toxic"],
}


def map_to_theme(keywords: list[str]) -> str:
    """Map BERTopic keywords to the closest pre-defined theme."""
    best_theme = "other"
    best_count = 0
    for theme, theme_keys in THEME_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in theme_keys)
        if count > best_count:
            best_count = count
            best_theme = theme
    return best_theme


def run_topic_modeling(min_topic_size: int = 5) -> pd.DataFrame:
    """
    Full BERTopic pipeline on sentiment-scored reviews.

    Args:
        min_topic_size: Minimum cluster size for HDBSCAN (lower = more topics).
    """
    os.makedirs("output", exist_ok=True)

    # Load sentiment results
    df = pd.read_csv(INPUT_CSV)
    docs = df["review_text"].tolist()
    print(f"Loaded {len(docs)} reviews for topic modeling.")

    # ── 1. Embed with sentence-transformers ───────────────────────────────────
    print("Embedding reviews (downloads model on first run)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")   # 80MB, fast

    # ── 2. Configure UMAP + HDBSCAN ──────────────────────────────────────────
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # ── 3. BERTopic ──────────────────────────────────────────────────────────
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    df["topic_id"] = topics
    df["topic_prob"] = [round(float(p), 4) if not np.isnan(p) else 0.0
                        for p in probs]

    # ── 4. Build topic summary ────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    print(f"\n✅  Found {len(topic_info) - 1} topics (excl. outlier topic -1)")

    topic_summary = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue   # skip outliers

        # Get top keywords
        keywords = [kw for kw, _ in topic_model.get_topic(tid)]

        # Map to human-readable theme
        theme = map_to_theme(keywords)

        # Compute average sentiment for this topic
        topic_reviews = df[df["topic_id"] == tid]
        avg_score = round(topic_reviews["final_score"].mean(), 2)
        sentiment_dist = topic_reviews["final_label"].value_counts().to_dict()

        topic_summary.append({
            "topic_id":       int(tid),
            "theme":          theme,
            "count":          int(row["Count"]),
            "keywords":       keywords[:8],
            "avg_sentiment":  avg_score,
            "sentiment_dist": sentiment_dist,
        })

    # Sort by count descending
    topic_summary.sort(key=lambda x: x["count"], reverse=True)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_json(OUTPUT_JSON, orient="records", indent=2)

    with open(TOPICS_JSON, "w") as f:
        json.dump(topic_summary, f, indent=2)

    print(f"✅  Topic results → {OUTPUT_CSV}")
    print(f"✅  Topic summary → {TOPICS_JSON}")

    # Print summary table
    print("\n── Topic Summary ──────────────────────────────────────────────────")
    for t in topic_summary:
        print(f"  Topic {t['topic_id']:2d} | {t['theme']:12s} | "
              f"count={t['count']:3d} | avg_sentiment={t['avg_sentiment']:.1f} | "
              f"keywords: {', '.join(t['keywords'][:4])}")

    return df, topic_summary


if __name__ == "__main__":
    df, summary = run_topic_modeling(min_topic_size=5)
    print(f"\nTotal reviews with topic assignment: {len(df)}")
    print(f"Reviews in outlier topic (-1): {(df['topic_id'] == -1).sum()}")
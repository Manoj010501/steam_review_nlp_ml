"""
trend_aggregation.py
--------------------
P3 Responsibility: Keyword + trend aggregation.

Computes:
  - Monthly sentiment trend (avg score per month per game)
  - Top keywords per month (TF-IDF based)
  - Sentiment velocity (how fast sentiment is changing)
  - Trending negative keywords (early warning signals for devs)

Output is consumed by the Frontend (P1) for the Trend Tracking charts.
"""

import pandas as pd
import numpy as np
import json
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Always run relative to THIS script's folder, not the terminal's cwd ───────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working directory: {os.getcwd()}")

OUTPUT_JSON = "output/trend_data.json"
OUTPUT_CSV  = "output/trend_data.csv"


# ── Smart loader: prefer topics_results, fall back to sentiment_results ────────

def load_data() -> pd.DataFrame:
    """
    Load the best available CSV.
    Priority: topics_results.csv → sentiment_results.csv → error
    """
    candidates = [
        "output/topics_results.csv",
        "output/sentiment_results.csv",
    ]

    for path in candidates:
        if not os.path.exists(path):
            print(f"  [skip] {path} — not found")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"  [skip] {path} — file is empty")
            continue

        print(f"  [✓] Loaded {len(df)} rows from {path}")
        print(f"  Columns: {list(df.columns)}")

        # Ensure required columns exist
        required = ["review_text", "game_name", "month", "final_label", "final_score"]
        missing = [c for c in required if c in df.columns]   # columns that exist
        absent  = [c for c in required if c not in df.columns]

        if absent:
            print(f"  [warn] Missing columns: {absent} — will patch with defaults")

        # Patch missing columns so the rest of the pipeline can run
        if "final_label" not in df.columns:
            # Fall back to vader_label if available
            if "vader_label" in df.columns:
                df["final_label"] = df["vader_label"]
                print("  [patch] final_label ← vader_label")
            else:
                df["final_label"] = "neutral"
                print("  [patch] final_label ← 'neutral' (no sentiment column found)")

        if "final_score" not in df.columns:
            if "vader_score" in df.columns:
                df["final_score"] = df["vader_score"]
                print("  [patch] final_score ← vader_score")
            else:
                df["final_score"] = 50.0
                print("  [patch] final_score ← 50.0 (default)")

        if "game_name" not in df.columns:
            df["game_name"] = "Unknown Game"

        if "month" not in df.columns:
            if "date" in df.columns:
                df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
                print("  [patch] month ← derived from date column")
            else:
                df["month"] = "2024-01"

        return df

    raise FileNotFoundError(
        "No input CSV found. Run sentiment_pipeline.py (Step 2) first.\n"
        "Expected: output/sentiment_results.csv"
    )


# ── Monthly Sentiment Trend ────────────────────────────────────────────────────

def compute_monthly_sentiment(df: pd.DataFrame) -> list[dict]:
    results = []
    for game_name, game_df in df.groupby("game_name"):
        for month, month_df in game_df.groupby("month"):
            label_counts = month_df["final_label"].value_counts().to_dict()
            total = len(month_df)
            results.append({
                "game_name":    game_name,
                "app_id":       str(month_df["app_id"].iloc[0]) if "app_id" in month_df else "",
                "month":        month,
                "avg_score":    round(float(month_df["final_score"].mean()), 2),
                "review_count": total,
                "positive":     label_counts.get("positive", 0),
                "negative":     label_counts.get("negative", 0),
                "neutral":      label_counts.get("neutral",  0),
                "positive_pct": round(label_counts.get("positive", 0) / max(total, 1) * 100, 1),
            })
    results.sort(key=lambda x: (x["game_name"], x["month"]))
    return results


# ── Sentiment Velocity ─────────────────────────────────────────────────────────

def compute_velocity(monthly: list[dict]) -> list[dict]:
    velocity_records = []
    by_game = {}
    for r in monthly:
        by_game.setdefault(r["game_name"], []).append(r)

    for game_name, records in by_game.items():
        records.sort(key=lambda x: x["month"])
        for i, rec in enumerate(records):
            velocity = 0.0 if i == 0 else round(
                rec["avg_score"] - records[i-1]["avg_score"], 2)
            velocity_records.append({
                **rec,
                "velocity": velocity,
                "trend": "improving" if velocity > 2 else
                         "declining" if velocity < -2 else "stable",
            })
    return velocity_records


# ── Keyword Extraction ─────────────────────────────────────────────────────────

def extract_monthly_keywords(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    keyword_records = []
    for game_name, game_df in df.groupby("game_name"):
        for month, month_df in game_df.groupby("month"):
            for sentiment in ["positive", "negative"]:
                seg = month_df[month_df["final_label"] == sentiment]
                if len(seg) < 3:
                    continue
                try:
                    tfidf = TfidfVectorizer(
                        stop_words="english", ngram_range=(1, 2), max_features=200)
                    tfidf.fit_transform(seg["review_text"].astype(str).tolist())
                    scores   = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
                    keywords = [kw for kw, _ in sorted(scores.items(), key=lambda x: x[1])[:top_n]]
                except Exception as e:
                    keywords = []
                keyword_records.append({
                    "game_name": game_name,
                    "month":     month,
                    "sentiment": sentiment,
                    "keywords":  keywords,
                })
    return keyword_records


# ── Trending Negative Keywords ─────────────────────────────────────────────────

def find_trending_negatives(df: pd.DataFrame, months_back: int = 3) -> list[dict]:
    neg_df = df[df["final_label"] == "negative"].copy()
    sorted_months = sorted(neg_df["month"].unique())
    recent_months = sorted_months[-months_back:]
    older_months  = sorted_months[:-months_back]

    def get_kw_counts(subset):
        if len(subset) < 2:
            return Counter()
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=100)
            tfidf.fit_transform(subset["review_text"].astype(str).tolist())
            return Counter({kw: 1 for kw in tfidf.get_feature_names_out()})
        except Exception:
            return Counter()

    recent = get_kw_counts(neg_df[neg_df["month"].isin(recent_months)])
    older  = get_kw_counts(neg_df[neg_df["month"].isin(older_months)])

    trending = [
        {"keyword": kw, "recent_count": cnt,
         "older_count": older.get(kw, 0), "is_new": older.get(kw, 0) == 0}
        for kw, cnt in recent.items() if cnt > older.get(kw, 0)
    ]
    trending.sort(key=lambda x: x["recent_count"], reverse=True)
    return trending[:20]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_trend_aggregation() -> dict:
    os.makedirs("output", exist_ok=True)

    print("\n── Loading data ──────────────────────────────────────────────────")
    df = load_data()

    print("\n── Computing monthly sentiment ───────────────────────────────────")
    monthly_raw  = compute_monthly_sentiment(df)
    monthly_vel  = compute_velocity(monthly_raw)
    print(f"  {len(monthly_vel)} month/game records")

    print("\n── Extracting keywords ───────────────────────────────────────────")
    keywords = extract_monthly_keywords(df)
    print(f"  {len(keywords)} keyword records")

    print("\n── Finding trending negatives ────────────────────────────────────")
    neg_trending = find_trending_negatives(df)
    print(f"  {len(neg_trending)} trending negative keywords")

    output = {
        "monthly_trend":      monthly_vel,
        "monthly_keywords":   keywords,
        "trending_negatives": neg_trending,
        "summary": {
            "total_reviews":     len(df),
            "games":             df["game_name"].unique().tolist(),
            "months":            sorted(df["month"].unique().tolist()),
            "overall_avg_score": round(float(df["final_score"].mean()), 2),
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    pd.DataFrame(monthly_vel).to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅  trend_data.json  → {OUTPUT_JSON}")
    print(f"✅  trend_data.csv   → {OUTPUT_CSV}")

    print("\n── Monthly Trend Preview ─────────────────────────────────────────")
    for r in monthly_vel[:8]:
        arrow = "↑" if r["velocity"] > 0 else ("↓" if r["velocity"] < 0 else "→")
        print(f"  {r['game_name'][:20]:20s}  {r['month']}  "
              f"score={r['avg_score']:5.1f}  "
              f"velocity={r['velocity']:+.1f} {arrow}  "
              f"({r['review_count']} reviews)")

    print("\n── Top Trending Negative Keywords ───────────────────────────────")
    for item in neg_trending[:5]:
        tag = " (NEW)" if item["is_new"] else ""
        print(f"  '{item['keyword']}'{tag}")

    print(f"\n── Summary ───────────────────────────────────────────────────────")
    s = output["summary"]
    print(f"  Total reviews : {s['total_reviews']}")
    print(f"  Games         : {', '.join(s['games'])}")
    print(f"  Months        : {s['months'][0]} → {s['months'][-1]}")
    print(f"  Avg score     : {s['overall_avg_score']}/100")

    return output


if __name__ == "__main__":
    run_trend_aggregation()
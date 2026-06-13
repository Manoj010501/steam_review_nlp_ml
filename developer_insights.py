"""
developer_insights.py
---------------------
P3 Responsibility: Actionable developer insights from negative review patterns.

Generates structured recommendations like:
  - "30% of negative reviews in the last 2 months mention 'crash' — investigate stability"
  - "Performance topic sentiment dropped 15 points since launch"

Output is fed into the Developer Insights panel (Frontend P1).
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

# ── Always run relative to this script folder ────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))



INPUT_TOPICS_CSV  = "output/topics_results.csv"
INPUT_TREND_JSON  = "output/trend_data.json"
OUTPUT_JSON       = "output/developer_insights.json"


# ── Insight generators ─────────────────────────────────────────────────────────

def insight_crash_bugs(df: pd.DataFrame) -> list[dict]:
    """Flag reviews mentioning crashes/bugs in recent months."""
    crash_keywords = ["crash", "bug", "freeze", "corrupt", "broken",
                      "unplayable", "error", "glitch", "stuck"]
    insights = []

    neg_df = df[df["final_label"] == "negative"]
    recent = sorted(df["month"].unique())[-3:]
    recent_neg = neg_df[neg_df["month"].isin(recent)]

    for kw in crash_keywords:
        matches = recent_neg[recent_neg["review_text"].str.lower().str.contains(kw)]
        if len(matches) == 0:
            continue
        pct = round(len(matches) / max(len(recent_neg), 1) * 100, 1)
        if pct >= 5:
            insights.append({
                "type":      "bug_report",
                "severity":  "high" if pct > 15 else "medium",
                "keyword":   kw,
                "count":     len(matches),
                "pct_of_neg": pct,
                "message":   (
                    f"{pct}% of recent negative reviews mention '{kw}'. "
                    f"Recommend investigating stability issues."
                ),
                "games_affected": matches["game_name"].unique().tolist(),
            })

    return insights


def insight_topic_decline(topic_summary: list[dict],
                          monthly_trend: list[dict]) -> list[dict]:
    """
    Compare topic sentiment now vs overall average.
    Flag topics that are significantly below average.
    """
    insights = []
    overall_avg = np.mean([r["avg_score"] for r in monthly_trend])

    for topic in topic_summary:
        if topic["avg_sentiment"] < overall_avg - 15:
            insights.append({
                "type":    "topic_decline",
                "severity": "high" if topic["avg_sentiment"] < 35 else "medium",
                "theme":    topic["theme"],
                "avg_sentiment": topic["avg_sentiment"],
                "overall_avg":   round(overall_avg, 1),
                "gap":     round(overall_avg - topic["avg_sentiment"], 1),
                "message": (
                    f"Topic '{topic['theme']}' has avg sentiment "
                    f"{topic['avg_sentiment']:.0f}/100, which is "
                    f"{overall_avg - topic['avg_sentiment']:.0f} points below "
                    f"the overall average. "
                    f"Top complaints: {', '.join(topic['keywords'][:4])}."
                ),
            })

    return insights


def insight_sentiment_velocity(monthly_trend: list[dict]) -> list[dict]:
    """Flag games with rapidly declining sentiment over last 2 months."""
    insights = []
    by_game = defaultdict(list)
    for r in monthly_trend:
        by_game[r["game_name"]].append(r)

    for game_name, records in by_game.items():
        records.sort(key=lambda x: x["month"])
        last_2 = records[-2:]
        if len(last_2) < 2:
            continue
        velocity = last_2[-1]["avg_score"] - last_2[0]["avg_score"]
        if velocity < -8:
            insights.append({
                "type":     "sentiment_decline",
                "severity": "high" if velocity < -15 else "medium",
                "game_name": game_name,
                "velocity": round(velocity, 1),
                "current_score": last_2[-1]["avg_score"],
                "message": (
                    f"{game_name} sentiment dropped {abs(velocity):.0f} points "
                    f"in the last 2 months (now {last_2[-1]['avg_score']:.0f}/100). "
                    f"Review recent patches and community feedback."
                ),
            })

    return insights


def insight_praise_patterns(df: pd.DataFrame) -> list[dict]:
    """What are players consistently praising? Good for marketing."""
    insights = []
    pos_df = df[df["final_label"] == "positive"]

    praise_keywords = {
        "visuals":     ["graphics", "beautiful", "stunning", "art", "visuals"],
        "story":       ["story", "narrative", "characters", "immersive", "plot"],
        "gameplay":    ["combat", "satisfying", "fun", "smooth", "controls"],
        "performance": ["fps", "optimized", "runs", "stable", "performance"],
        "value":       ["worth", "price", "hours", "content", "value"],
    }

    for category, keywords in praise_keywords.items():
        matches = pos_df[
            pos_df["review_text"].str.lower()
                .str.contains("|".join(keywords))
        ]
        pct = round(len(matches) / max(len(pos_df), 1) * 100, 1)
        if pct >= 15:
            insights.append({
                "type":     "praise_signal",
                "severity": "info",
                "category": category,
                "count":    len(matches),
                "pct_of_pos": pct,
                "message":  (
                    f"{pct}% of positive reviews praise '{category}'. "
                    f"Consider highlighting this in marketing materials."
                ),
            })

    return insights


# ── Assemble all insights ──────────────────────────────────────────────────────

def run_developer_insights() -> list[dict]:
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv(INPUT_TOPICS_CSV)

    with open(INPUT_TREND_JSON) as f:
        trend_data = json.load(f)

    monthly_trend  = trend_data["monthly_trend"]
    topic_summary  = []    # will be loaded from topic_summary.json if available
    try:
        with open("output/topic_summary.json") as f:
            topic_summary = json.load(f)
    except FileNotFoundError:
        pass

    # Run all insight generators
    all_insights = (
        insight_crash_bugs(df) +
        insight_topic_decline(topic_summary, monthly_trend) +
        insight_sentiment_velocity(monthly_trend) +
        insight_praise_patterns(df)
    )

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "info": 2}
    all_insights.sort(key=lambda x: severity_order.get(x["severity"], 3))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_insights, f, indent=2)

    print(f"✅  Developer insights → {OUTPUT_JSON}")
    print(f"\n── {len(all_insights)} Insights Generated ───────────────────────")
    for ins in all_insights:
        icon = "🔴" if ins["severity"] == "high" else \
               "🟡" if ins["severity"] == "medium" else "🟢"
        print(f"  {icon} [{ins['type']}] {ins['message'][:80]}...")

    return all_insights


if __name__ == "__main__":
    run_developer_insights()

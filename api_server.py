"""
api_server.py
-------------
P3 delivers data to P2 (Backend/API Dev) via these Flask endpoints.
P2 will proxy these or merge them into their main Flask app.

Endpoints:
  GET /api/sentiment?app_id=XXXX        → sentiment results for a game
  GET /api/topics?app_id=XXXX           → topic clusters for a game
  GET /api/trends?app_id=XXXX           → monthly trend data
  GET /api/insights?app_id=XXXX         → developer insights
  GET /api/summary                      → overall summary stats
  POST /api/analyze                     → analyze a batch of new reviews (live)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
import os

from sentiment_pipeline import VADERScorer

app = Flask(__name__)
CORS(app)

# ── Load pre-computed outputs ──────────────────────────────────────────────────

def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def filter_by_app(data: list[dict], app_id: str | None) -> list[dict]:
    if not app_id:
        return data
    return [r for r in data if str(r.get("app_id", "")) == app_id]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "module": "NLP/ML P3"})


@app.route("/api/sentiment")
def get_sentiment():
    app_id = request.args.get("app_id")
    df = load_csv("output/sentiment_results.csv")
    if df is None:
        return jsonify({"error": "Run sentiment_pipeline.py first"}), 404

    if app_id:
        df = df[df["app_id"].astype(str) == app_id]

    records = df[[
        "review_id", "app_id", "game_name", "review_text",
        "month", "playtime_hours",
        "vader_label", "vader_score",
        "final_label", "final_score",
    ]].to_dict(orient="records")

    # Aggregate stats
    stats = {
        "total":    len(df),
        "positive": int((df["final_label"] == "positive").sum()),
        "negative": int((df["final_label"] == "negative").sum()),
        "neutral":  int((df["final_label"] == "neutral").sum()),
        "avg_score": round(float(df["final_score"].mean()), 2),
    }

    return jsonify({"stats": stats, "reviews": records})


@app.route("/api/topics")
def get_topics():
    app_id = request.args.get("app_id")

    topic_summary = load_json("output/topic_summary.json") or []
    df = load_csv("output/topics_results.csv")

    if df is None:
        return jsonify({"error": "Run topic_modeling.py first"}), 404

    if app_id:
        df = df[df["app_id"].astype(str) == app_id]

    # Rebuild topic summary filtered to this game
    filtered_topics = []
    for t in topic_summary:
        topic_reviews = df[df["topic_id"] == t["topic_id"]]
        if len(topic_reviews) == 0:
            continue
        t_copy = dict(t)
        t_copy["count"] = len(topic_reviews)
        t_copy["avg_sentiment"] = round(
            float(topic_reviews["final_score"].mean()), 2)
        filtered_topics.append(t_copy)

    return jsonify({"topics": filtered_topics})


@app.route("/api/trends")
def get_trends():
    app_id = request.args.get("app_id")
    data   = load_json("output/trend_data.json")
    if data is None:
        return jsonify({"error": "Run trend_aggregation.py first"}), 404

    monthly = filter_by_app(data["monthly_trend"], app_id)
    keywords = [
        r for r in data["monthly_keywords"]
        if not app_id or r.get("game_name") in
            {m["game_name"] for m in monthly}
    ]

    return jsonify({
        "monthly_trend":      monthly,
        "monthly_keywords":   keywords,
        "trending_negatives": data["trending_negatives"],
        "summary":            data["summary"],
    })


@app.route("/api/insights")
def get_insights():
    insights = load_json("output/developer_insights.json")
    if insights is None:
        return jsonify({"error": "Run developer_insights.py first"}), 404

    app_id = request.args.get("app_id")
    severity = request.args.get("severity")   # optional filter

    filtered = insights
    if severity:
        filtered = [i for i in filtered if i.get("severity") == severity]

    return jsonify({"insights": filtered, "total": len(filtered)})


@app.route("/api/summary")
def get_summary():
    """Overall stats across all games — for the dashboard header."""
    df = load_csv("output/sentiment_results.csv")
    if df is None:
        return jsonify({"error": "Run sentiment_pipeline.py first"}), 404

    return jsonify({
        "total_reviews": len(df),
        "games": df["game_name"].unique().tolist(),
        "avg_score": round(float(df["final_score"].mean()), 2),
        "sentiment_dist": df["final_label"].value_counts().to_dict(),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_live():
    """
    Analyze a batch of new review texts in real time (no DB).
    Body: { "reviews": [{"review_id": "...", "text": "..."}] }
    Returns sentiment for each review.
    """
    body = request.json
    if not body or "reviews" not in body:
        return jsonify({"error": "Send JSON with 'reviews' list"}), 400

    vader = VADERScorer()
    results = []
    for item in body["reviews"]:
        score = vader.score(item.get("text", ""))
        results.append({
            "review_id":   item.get("review_id", ""),
            "vader_label": score["vader_label"],
            "vader_score": score["vader_score"],
            # BERT skipped for live endpoint to keep latency low
        })

    return jsonify({"results": results})


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting NLP/ML API server on http://localhost:5001")
    print("Endpoints: /api/health  /api/sentiment  /api/topics")
    print("           /api/trends  /api/insights   /api/summary")
    print("           /api/analyze (POST)")
    app.run(host="0.0.0.0", port=5001, debug=True)

# Steam Game Review Analytics Dashboard
## NLP / ML Pipeline (P3)

This module is responsible for turning raw Steam game reviews into structured sentiment scores, topic clusters, trend data, and developer insights. It is one of four components in the overall Steam Game Review Analytics Dashboard project.

---

## Role & Responsibilities

As the **NLP/ML Developer (P3)**, my scope covers everything between raw review text and the structured data consumed by the **Frontend (P1)** and **Backend (P2)** teams:

| Responsibility | Description |
|---|---|
| Sentiment Scoring Pipeline | VADER + BERT ensemble classifies each review as positive, negative, or neutral with a 0–100 confidence score |
| Topic Cluster Extraction | BERTopic groups reviews into themes (gameplay, performance, story, value, multiplayer, developer) using sentence embeddings + HDBSCAN |
| Keyword & Trend Aggregation | Month-by-month sentiment curves, TF-IDF keyword extraction, and sentiment velocity per game |
| Developer Insights Engine | Auto-generates actionable recommendations from recurring negative-review patterns and topic-level sentiment decline |
| Model Accuracy Evaluation | Accuracy, precision/recall/F1, and confusion matrix comparing VADER, BERT, and the ensemble against ground-truth labels |

---

## Tech Stack

- **NLP / ML Core:** vaderSentiment, BERTopic, sentence-transformers, scikit-learn, UMAP + HDBSCAN
- **Data Processing:** pandas, numpy, TF-IDF vectorization
- **Testing:** pytest (24 tests)
- **Integration:** Flask REST API (port 5001)

---

## How It Works

### What is VADER?
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analyzer. It scores text by matching words against a pre-built sentiment dictionary and applying rules for negation, intensifiers, and punctuation. It is fast and requires no training, but has limited understanding of context.

### What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a deep learning language model that reads entire sentences for context, rather than scoring word-by-word. This project uses a lightweight, pre-trained variant (`distilbert-base-uncased-finetuned-sst-2-english`) fine-tuned for sentiment classification.

### Why combine them?
VADER is fast and interpretable but shallow. BERT understands context better but can be overconfident or unfamiliar with gaming-specific terms. The pipeline blends both (BERT weighted 65%, VADER 35%) into a final ensemble score for more reliable results.

---

## Pipeline Architecture

The pipeline runs as six sequential stages. Each stage reads the previous stage's output and writes new files to `output/`.

```
python run_pipeline.py
```

| Step | Script | Output |
|---|---|---|
| 1 | `generate_sample_data.py` | `data/sample_reviews.csv` — 300 synthetic reviews across 5 games |
| 2 | `sentiment_pipeline.py` | `output/sentiment_results.csv` — VADER + BERT ensemble scores |
| 3 | `topic_modeling.py` | `output/topics_results.csv`, `topic_summary.json` — 19 topic clusters |
| 4 | `trend_aggregation.py` | `output/trend_data.json`, `trend_data.csv` — monthly sentiment & keyword trends |
| 5 | `developer_insights.py` | `output/developer_insights.json` — auto-generated recommendations |
| 6 | `evaluate_model.py` | `output/evaluation_report.json` — accuracy, F1, confusion matrix |

---

## Results (Sample Data, n=300)

### Sentiment Distribution
| Sentiment | Count | Percentage |
|---|---|---|
| Positive | 183 | 61% |
| Negative | 89 | 30% |
| Neutral | 28 | 9% |

Overall average sentiment score: **59.7 / 100**

### Topic Modeling
19 topic clusters extracted, mapped to 6 themes:

| Theme | Review Count | Avg Sentiment |
|---|---|---|
| Performance | 39 | 79.1 |
| Value | 30 | 66.0 |
| Gameplay | 74 | 63.3 |
| Developer | 44 | 59.2 |
| Story | 23 | 52.2 |
| Multiplayer | 31 | 42.0 |

**Key finding:** Performance-related feedback is the most positively received theme; multiplayer issues (mainly server downtime) drag the lowest average sentiment.

### Model Evaluation
| Metric | Value |
|---|---|
| Overall Accuracy | 79.3% |
| F1 Macro Score | 63.0% |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Positive | 83.1% | 93.2% | 87.9% |
| Negative | 91.0% | 81.0% | 85.7% |
| Neutral | 17.9% | 13.5% | 15.4% |

**Known limitation:** VADER's rule-based scoring tends to push ambiguous "neutral" reviews toward positive or negative. A fine-tuned 3-class transformer trained on real Steam review data would likely improve neutral-class separation.

---

## Project Structure

```
nlp_ml/
├── generate_sample_data.py    # Step 1: sample data generator
├── sentiment_pipeline.py      # Step 2: VADER + BERT sentiment scoring
├── topic_modeling.py          # Step 3: BERTopic cluster extraction
├── trend_aggregation.py       # Step 4: monthly trends + keywords
├── developer_insights.py      # Step 5: auto-generated recommendations
├── evaluate_model.py          # Step 6: accuracy evaluation
├── api_server.py              # Flask API — hand-off point to P2
├── run_pipeline.py            # Orchestrates all 6 steps
├── data/
│   └── sample_reviews.csv
├── output/
│   └── *.csv, *.json          # All pipeline outputs
└── tests/
    └── test_pipeline.py       # 24 pytest tests
```

---

## Setup & Usage

### 1. Create environment
```bash
conda create -n nlpml python=3.11 -y
conda activate nlpml
conda install -y numpy=1.26.4 scipy=1.11.4 scikit-learn=1.4.2 pandas=2.2.2
conda install -y -c conda-forge hdbscan umap-learn
pip install -r requirements_pip.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. Start the API server (for P2 integration)
```bash
python api_server.py
```
Runs on `http://localhost:5001`

---

## API Endpoints (for P2)

| Endpoint | Method | Description |
|---|---|---|
| `/api/sentiment?app_id=<id>` | GET | Sentiment results for a game |
| `/api/topics?app_id=<id>` | GET | Topic clusters with sentiment per theme |
| `/api/trends?app_id=<id>` | GET | Monthly sentiment trends and keywords |
| `/api/insights` | GET | Developer recommendations |
| `/api/summary` | GET | Overall dashboard stats |
| `/api/analyze` | POST | Real-time sentiment scoring for new reviews |

---

## Hand-off to Other Team mates

- **P1 (Frontend):** `topic_summary.json`, `trend_data.json`, `developer_insights.json` — ready for Recharts visualizations
- **P2 (Backend):** `sentiment_results.csv`, `topics_results.csv` — served via Flask API, or consumed directly as files

---

## Current Status & Next Steps

**Status:** Pipeline fully built and tested end-to-end on 300 synthetic sample reviews.

**Blocked on:** Real Steam review data from P2's Steam Web API integration.

**Next steps once real data arrives:**
1. Replace `data/sample_reviews.csv` with real review data (same column schema)
2. Re-run pipeline steps 2–6 (step 1 no longer needed)
3. Enable BERT scoring (`--bert` flag) and consider fine-tuning for improved neutral-class detection

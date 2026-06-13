"""
run_pipeline.py
---------------
Master orchestrator for the full P3 NLP/ML pipeline.

Run order:
  1. generate_sample_data   → data/sample_reviews.csv
  2. sentiment_pipeline     → output/sentiment_results.csv
  3. topic_modeling         → output/topics_results.csv + topic_summary.json
  4. trend_aggregation      → output/trend_data.json
  5. developer_insights     → output/developer_insights.json
  6. evaluate_model         → output/evaluation_report.json

Usage:
  python run_pipeline.py              # full pipeline (VADER only, fast)
  python run_pipeline.py --bert       # full pipeline with BERT (slower)
  python run_pipeline.py --step 2     # run only step 2 (sentiment)
"""

import argparse
import time
import sys

BANNER = """
╔══════════════════════════════════════════════════════╗
║         Steam Review Analytics — P3 Pipeline        ║
║         NLP / ML Developer                          ║
╚══════════════════════════════════════════════════════╝
"""

def run_step(name: str, fn, *args, **kwargs):
    print(f"\n{'─'*55}")
    print(f"  STEP: {name}")
    print(f"{'─'*55}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  ✅  Done in {elapsed:.1f}s")
    return result


def main():
    print(BANNER)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", action="store_true",
                        help="Enable BERT scoring (slower, more accurate)")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (1-6)")
    args = parser.parse_args()

    # ── Import after parse so imports don't fail on --help ───────────────────
    from generate_sample_data import generate_reviews
    from sentiment_pipeline   import run_sentiment_pipeline
    from topic_modeling       import run_topic_modeling
    from trend_aggregation    import run_trend_aggregation
    from developer_insights   import run_developer_insights
    from evaluate_model       import run_evaluation

    steps = {
        1: ("Generate sample data",    generate_reviews,         {}),
        2: ("Sentiment pipeline",      run_sentiment_pipeline,   {"use_bert": args.bert}),
        3: ("Topic modeling",          run_topic_modeling,       {"min_topic_size": 5}),
        4: ("Trend aggregation",       run_trend_aggregation,    {}),
        5: ("Developer insights",      run_developer_insights,   {}),
        6: ("Model evaluation",        run_evaluation,           {}),
    }

    if args.step:
        if args.step not in steps:
            print(f"Unknown step {args.step}. Valid: 1-6")
            sys.exit(1)
        name, fn, kwargs = steps[args.step]
        run_step(name, fn, **kwargs)
    else:
        for num in sorted(steps):
            name, fn, kwargs = steps[num]
            run_step(name, fn, **kwargs)

    print(f"\n{'═'*55}")
    print("  ALL STEPS COMPLETE — output/ folder ready for P2/P1")
    print(f"{'═'*55}\n")
    print("Output files:")
    import os
    for f in sorted(os.listdir("output")):
        size = os.path.getsize(f"output/{f}")
        print(f"  output/{f:35s} ({size:>7,} bytes)")


if __name__ == "__main__":
    main()

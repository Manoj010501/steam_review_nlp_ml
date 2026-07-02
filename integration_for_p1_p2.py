"""
integration_for_p1_p2.py
------------------------
Code snippets for P2 (Backend/Streamlit) to integrate P3's NLP/ML
analytics pipeline into the dashboard.

Instructions for P2:
1. Clone P3's repo: https://github.com/Manoj010501/steam_review_nlp_ml
2. Add the refresh button code to your Streamlit dashboard file
3. Add the analytics loader code wherever you want to display
   topics, trends, and insights
4. Update the path in both snippets to match your local machine
"""

# ══════════════════════════════════════════════════════════════════
# SNIPPET 1: Refresh Button (add to your Streamlit dashboard)
# ══════════════════════════════════════════════════════════════════

"""
import subprocess
import os

st.sidebar.title("Analytics Controls")

if st.sidebar.button("🔄 Refresh Analytics Data"):
    progress = st.sidebar.empty()
    
    steps = [
        ("Fetching latest Steam reviews...", "fetch_real_data.py"),
        ("Running sentiment analysis...",    "sentiment_pipeline.py"),
        ("Running topic modeling...",        "topic_modeling.py"),
        ("Computing trends...",              "trend_aggregation.py"),
        ("Generating insights...",           "developer_insights.py"),
    ]
    
    success = True
    for message, script in steps:
        progress.info(f"⏳ {message}")
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            cwd=r"D:\nlp_ml"  # ← CHANGE THIS to your local path
        )
        if result.returncode != 0:
            progress.error(f"❌ Failed at: {script}\n{result.stderr}")
            success = False
            break
    
    if success:
        progress.success("✅ Analytics updated with latest Steam reviews!")
        st.rerun()
"""

# ══════════════════════════════════════════════════════════════════
# SNIPPET 2: Analytics Loader (add wherever you display analytics)
# ══════════════════════════════════════════════════════════════════

"""
import json
import os

@st.cache_data(ttl=300)
def load_analytics():
    base = r"D:\nlp_ml\output"  # ← CHANGE THIS to your local path
    try:
        topics   = json.load(open(os.path.join(base, "topic_summary.json")))
        trends   = json.load(open(os.path.join(base, "trend_data.json")))
        insights = json.load(open(os.path.join(base, "developer_insights.json")))
        return topics, trends, insights
    except FileNotFoundError:
        return None, None, None

topics, trends, insights = load_analytics()

if topics is None:
    st.warning("Analytics data not found. Click Refresh Analytics Data to generate.")
else:
    st.subheader("Topic Analysis")
    st.json(topics)
    
    st.subheader("Monthly Trends")
    st.json(trends)
    
    st.subheader("Developer Insights")
    st.json(insights)
"""
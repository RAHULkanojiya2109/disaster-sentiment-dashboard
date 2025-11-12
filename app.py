import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- Optional imports with safe fallbacks ----------------
# WordCloud import (optional)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# NLTK import + safe setup
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    def ensure_nltk_data():
        resources = {
            "vader_lexicon": "sentiment/vader_lexicon",
            "punkt": "tokenizers/punkt",
            "stopwords": "corpora/stopwords",
            "wordnet": "corpora/wordnet",
            "omw-1.4": "corpora/omw-1.4"
        }
        for name, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                # download quietly; may take time on first run
                nltk.download(name, quiet=True)

    # Try to ensure resources (non-fatal)
    try:
        ensure_nltk_data()
        sia = SentimentIntensityAnalyzer()
        NLTK_AVAILABLE = True
    except Exception:
        # If anything fails here, still attempt to create SIA (may raise later)
        try:
            sia = SentimentIntensityAnalyzer()
            NLTK_AVAILABLE = True
        except Exception:
            NLTK_AVAILABLE = False
            sia = None

except Exception:
    NLTK_AVAILABLE = False
    sia = None

# If nltk not available, create a dummy SIA to avoid crashes (returns neutral)
if not NLTK_AVAILABLE:
    class _DummySIA:
        def polarity_scores(self, t):
            return {"compound": 0.0}
    sia = _DummySIA()

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Disaster Sentiment Dashboard", layout="wide")
st.title("🏆 Social Media & Sentiment Analysis for Disaster Management")

# If NLTK wasn't available, show a small warning (non-blocking)
if not NLTK_AVAILABLE:
    st.warning("NLTK not available on this environment — sentiment scores are neutral. Add 'nltk' to requirements and redeploy for real sentiment.")

# ---------------- Data loader ----------------
@st.cache_data
def load_data(default_path: str = "data/train.csv") -> pd.DataFrame:
    """
    Expects Kaggle 'Real or Not? NLP with Disaster Tweets' format:
    columns: id, keyword, location, text, target
    """
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    uploaded = st.file_uploader(
        "Upload a CSV with columns: id, keyword, location, text, target",
        type=["csv"]
    )
    if uploaded is not None:
        return pd.read_csv(uploaded)
    st.info("Place `train.csv` in ./data or upload it above to continue.")
    st.stop()

df = load_data()

# Validate columns
required_cols = {"text"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV missing required column(s): {required_cols - set(df.columns)}")
    st.stop()

# ---------------- Cleaning ----------------
def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\S+|#\S+", " ", text)   # remove links/mentions/hashtags
    text = re.sub(r"[^A-Za-z\s]", " ", text)        # keep letters/spaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

df["clean_text"] = df["text"].astype(str).apply(basic_clean)

# ---------------- Sentiment ----------------
def label_from_compound(c):
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    return "Neutral"

@st.cache_data
def score_sentiment(texts: pd.Series) -> pd.DataFrame:
    # Use sia (either real or dummy) to compute compound
    scores = texts.apply(lambda t: float(sia.polarity_scores(t)["compound"]))
    labels = scores.apply(label_from_compound)
    return pd.DataFrame({"compound": scores, "sentiment": labels})

sent = score_sentiment(df["clean_text"])
df = pd.concat([df, sent], axis=1)

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
kw_options = sorted([k for k in df.get("keyword", pd.Series([], dtype=str)).dropna().unique()])
selected_kw = st.sidebar.multiselect("Keyword", kw_options)

if selected_kw:
    df_view = df[df["keyword"].isin(selected_kw)]
else:
    df_view = df.copy()

# ---------------- KPI cards ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tweets", f"{len(df_view):,}")
col2.metric("Negative", (df_view["sentiment"]=="Negative").sum())
col3.metric("Neutral",  (df_view["sentiment"]=="Neutral").sum())
col4.metric("Positive", (df_view["sentiment"]=="Positive").sum())

st.divider()

# ---------------- Charts ----------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment Distribution")
    try:
        fig = px.pie(df_view, names="sentiment", title="Overall Sentiment Share")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to draw pie chart: {e}")

with c2:
    st.subheader("Most Frequent Words")
    wc_text = " ".join(df_view["clean_text"].tolist())
    if wc_text.strip():
        if WORDCLOUD_AVAILABLE:
            try:
                wc = WordCloud(width=1000, height=500, background_color="white").generate(wc_text)
                st.image(wc.to_array(), use_column_width=True)
            except Exception as e:
                st.error(f"WordCloud generation failed: {e}")
        else:
            st.info("WordCloud package not available. Add 'wordcloud' to requirements to see it.")
    else:
        st.info("No text available for wordcloud (after filters).")

# If Kaggle dataset has 'target' (1=disaster, 0=not), show comparison
if "target" in df_view.columns:
    st.subheader("Target vs Sentiment (Kaggle labels vs our model)")
    try:
        fig2 = px.histogram(
            df_view,
            x="sentiment",
            color=df_view["target"].map({1:"Disaster",0:"Not Disaster"}),
            barmode="group",
            text_auto=True
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to draw histogram: {e}")

st.caption("Tip: use the sidebar to filter by keyword; drop your CSV in /data for auto-load.")

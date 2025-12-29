# =================== IMPORTS ===================
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from serpapi import GoogleSearch

nltk.download("punkt")

# =================== API KEYS ===================
SERP_API_KEY = "8198373a9102fdb800c25e0c8337ff05cfce241afeb057f3d5a276588fee86dd"

# =================== FUNCTIONS ===================
def sentence_similarity(claim, sentence):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([claim, sentence])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def detect_stance(sentence):
    sentence = sentence.lower()
    if any(w in sentence for w in ["deny", "denied", "false", "not"]):
        return "Contradicts"
    if any(w in sentence for w in ["confirm", "confirmed", "announced", "expanding", "is", "will"]):
        return "Supports"
    return "Unverified"

def detect_sentiment(text):
    text = text.lower()
    pos = ["increase", "growth", "confirmed", "expanding", "investment"]
    neg = ["deny", "false", "shutting", "misinformation"]
    score = sum(w in text for w in pos) - sum(w in text for w in neg)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

def ethical_warning(text):
    text = text.lower()
    clickbait_words = ["shocking", "you won't believe", "amazing", "secret", "must see", "breaking", "urgent"]
    extreme_words = ["disaster", "catastrophic", "fake", "crisis"]
    manipulative_words = ["must see", "urgent", "breaking"]
    warnings = []
    if any(w in text for w in clickbait_words):
        warnings.append("Clickbait")
    if any(w in text for w in extreme_words):
        warnings.append("Extreme")
    if any(w in text for w in manipulative_words):
        warnings.append("Manipulative")
    return ", ".join(warnings) if warnings else "None"

def final_verdict(results):
    support = sum(r["Stance"] == "Supports" for r in results)
    contradict = sum(r["Stance"] == "Contradicts" for r in results)
    total = len(results)
    if support > contradict:
        return "Verified", round(support / total, 2)
    elif contradict > support:
        return "Contradicted", round(contradict / total, 2)
    else:
        return "Unverified", 0.5

def fetch_serpapi_news(claim, num_results=10):
    params = {
        "engine": "google_news",
        "q": claim,
        "api_key": SERP_API_KEY,
        "num": num_results,
        "hl": "en",
        "gl": "us"
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    articles = []
    for item in results.get("news_results", []):
        articles.append({
            "source": item.get("source", "Unknown"),
            "date": item.get("date", "N/A"),
            "text": (item.get("title") or "") + ". " + (item.get("snippet") or "")
        })
    return articles

# =================== STREAMLIT UI ===================
st.title("📰 News Verifier & Context Analyzer")

claim = st.text_input(
    "Enter a news headline or claim:",
    placeholder="e.g. Amazon is shutting down data centers in Europe"
)

if st.button("Verify News"):

    if not claim.strip():
        st.warning("Please enter a claim first.")
        st.stop()

    st.info("Fetching articles...")

    raw_articles = fetch_serpapi_news(claim)

    if not raw_articles:
        st.warning("No articles found. Using fallback.")
        raw_articles = [{
            "source": "BBC",
            "date": "2024-11-02",
            "text": "Amazon announced it is expanding its data centers in Europe."
        }]

    # ================= RANKING =================
    ranked_articles = []

    for article in raw_articles:
        sentences = sent_tokenize(article["text"])
        max_score = 0
        for s in sentences:
            score = sentence_similarity(claim, s)
            max_score = max(max_score, score)

        article["rank_score"] = round(max_score, 3)
        ranked_articles.append(article)

    articles = sorted(
        ranked_articles,
        key=lambda x: x["rank_score"],
        reverse=True
    )[:5]

    # 🔍 RAW DATA VIEW
    with st.expander("🔍 View Ranked Articles (Top 5)"):
        st.dataframe(
            pd.DataFrame(articles)[["source", "date", "rank_score", "text"]]
        )

    # ================= ANALYSIS =================
    results = []
    for article in articles:
        sentences = sent_tokenize(article["text"])
        best_sentence = ""
        best_score = 0

        for s in sentences:
            score = sentence_similarity(claim, s)
            if score > best_score:
                best_score = score
                best_sentence = s

        results.append({
            "Source": article["source"],
            "Date": article["date"],
            "Relevance": round(best_score, 2),
            "Stance": detect_stance(best_sentence),
            "Sentiment": detect_sentiment(article["text"]),
            "Warnings": ethical_warning(article["text"]),
            "Evidence": best_sentence
        })

    # 📊 RESULTS
    st.subheader("📊 Verification Table")
    st.dataframe(pd.DataFrame(results))

    verdict, confidence = final_verdict(results)
    st.subheader("✅ Final Verdict")
    st.write(f"**Verdict:** {verdict}")
    st.write(f"**Confidence:** {confidence}")

    st.subheader("🕒 News Timeline")
    for r in sorted(results, key=lambda x: x["Date"]):
        st.write(f"{r['Date']} | {r['Source']} | {r['Stance']}")

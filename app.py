# =================== IMPORTS ===================
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from serpapi import GoogleSearch
import requests
import json

# nltk.download("punkt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# =================== API KEYS ===================
SERP_API_KEY = "8198373a9102fdb800c25e0c8337ff05cfce241afeb057f3d5a276588fee86dd"

# =================== LLM ===================
def run_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"]

def build_prompt(claim, articles):
    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"\nArticle {i+1}:\n{article}\n"

    return f"""
You are an expert fact-checking AI.

Claim:
"{claim}"

Articles:
{articles_text}

Return ONLY valid JSON:
{{
  "article_analysis": [
    {{
      "article_id": 1,
      "stance": "",
      "sentiment": "",
      "notes": ""
    }}
  ],
  "overall_warnings": "",
  "final_verdict": "",
  "confidence": 0
}}
"""

# =================== NLP HELPERS ===================
def sentence_similarity(claim, sentence):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([claim, sentence])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def detect_stance(sentence):
    s = sentence.lower()
    if any(w in s for w in ["deny", "denied", "false", "not"]):
        return "Contradicts"
    if any(w in s for w in ["confirm", "confirmed", "announced", "expanding", "will"]):
        return "Supports"
    return "Unverified"

def detect_sentiment(text):
    pos = ["increase", "growth", "confirmed", "expanding", "investment"]
    neg = ["deny", "false", "shutting", "misinformation"]
    score = sum(w in text.lower() for w in pos) - sum(w in text.lower() for w in neg)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

def ethical_warning(text):
    words = text.lower()
    warnings = []
    if any(w in words for w in ["shocking", "you won't believe", "breaking"]):
        warnings.append("Clickbait")
    if any(w in words for w in ["fake", "crisis", "disaster"]):
        warnings.append("Extreme")
    return ", ".join(warnings) if warnings else "None"

def final_verdict(results):
    support = sum(r["Stance"] == "Supports" for r in results)
    contradict = sum(r["Stance"] == "Contradicts" for r in results)
    total = len(results)
    if support > contradict:
        return "Verified", round(support / total, 2)
    elif contradict > support:
        return "Contradicted", round(contradict / total, 2)
    return "Unverified", 0.5

# =================== SERPAPI ===================
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
            "text": f"{item.get('title','')}. {item.get('snippet','')}"
        })
    return articles

def fetch_and_rank_articles(claim):
    raw_articles = fetch_serpapi_news(claim)

    if not raw_articles:
        raw_articles = [{
            "source": "BBC",
            "date": "2024-11-02",
            "text": "Amazon announced it is expanding its data centers in Europe."
        }]

    ranked = []
    for article in raw_articles:
        scores = [
            sentence_similarity(claim, s)
            for s in sent_tokenize(article["text"])
        ]
        article["rank_score"] = round(max(scores), 3)
        ranked.append(article)

    return sorted(ranked, key=lambda x: x["rank_score"], reverse=True)[:5]

# =================== STREAMLIT UI ===================
st.title("📰 News Verifier & Context Analyzer")

claim = st.text_input(
    "Enter a news headline or claim:",
    placeholder="Amazon is shutting down data centers in Europe"
)

col1, col2 = st.columns(2)

# ---------- VERIFY NEWS ----------
if col1.button("✅ Verify News"):
    if not claim.strip():
        st.warning("Please enter a claim.")
        st.stop()

    st.session_state.articles = fetch_and_rank_articles(claim)
    articles = st.session_state.articles

    results = []
    for article in articles:
        sentences = sent_tokenize(article["text"])
        best_sentence = max(sentences, key=lambda s: sentence_similarity(claim, s))

        results.append({
            "Source": article["source"],
            "Date": article["date"],
            "Relevance": article["rank_score"],
            "Stance": detect_stance(best_sentence),
            "Sentiment": detect_sentiment(article["text"]),
            "Warnings": ethical_warning(article["text"]),
            "Evidence": best_sentence
        })

    st.subheader("📊 Verification Results")
    st.dataframe(pd.DataFrame(results))

    verdict, confidence = final_verdict(results)
    st.subheader("✅ Final Verdict")
    st.write(f"**{verdict}** (confidence: {confidence})")

# ---------- LLM ANALYSIS ----------
if col2.button("🤖 Analysis through LLM"):
    if not claim.strip():
        st.warning("Please enter a claim.")
        st.stop()

    articles = fetch_and_rank_articles(claim)

    with st.spinner("Running LLM reasoning..."):
        prompt = build_prompt(claim, [a["text"] for a in articles])
        output = run_ollama(prompt)

    st.subheader("🧠 LLM Analysis")
    try:
        st.json(json.loads(output))
    except Exception:
        st.write(output)

    with st.expander("🔍 Ranked Articles Used"):
        st.dataframe(pd.DataFrame(articles)[["source", "date", "rank_score", "text"]])

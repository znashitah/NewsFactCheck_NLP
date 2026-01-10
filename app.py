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
from sentence_transformers import SentenceTransformer
import numpy as np
#Initialize embedding model:
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# nltk.download("punkt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# =================== API KEYS ===================
SERP_API_KEY = "8198373a9102fdb800c25e0c8337ff05cfce241afeb057f3d5a276588fee86dd"

# =================== LLM ===================

#def run_ollama(prompt):
#    response = requests.post(
#        OLLAMA_URL,
#        json={
#            "model": MODEL,
#            "prompt": prompt,
#            "stream": False
#        },
#        timeout=180
#    )
#    response.raise_for_status()
#    return response.json()["response"]

def run_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"    # 🚀 FORCE VALID JSON
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

IMPORTANT RULES (FOLLOW STRICTLY):
- You MUST verify the claim ONLY against the provided article texts.
- Do NOT use any external knowledge.
- Do NOT reinterpret or soften meanings.
- You MUST use ONLY these stance labels:
  - "Supports Claim"
  - "Refutes Claim"
  - "Neutral"

Claim:
"{claim}"

Articles:
{articles_text}

TASKS:
1. For EACH article, determine whether it SUPPORTS or REFUTES the claim.
2. After analyzing all articles:
   - If MOST articles refute the claim → final_verdict MUST say:
     "The claim is not supported by the articles."
   - If MOST articles support the claim → final_verdict MUST say:
     "The claim is supported by the articles."

CONFIDENCE RULE:
- Confidence is a number from 0 to 10.
- Confidence = percentage of articles agreeing with the final verdict.


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
if col1.button("Verify Claim via Classical NLP"):
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
if col2.button("Verify Claim through LLM"):
    if not claim.strip():
        st.warning("Please enter a claim.")
        st.stop()

    articles = fetch_and_rank_articles(claim)

    with st.spinner("Running LLM reasoning..."):
        prompt = build_prompt(claim, [a["text"] for a in articles])
        print(prompt)
        output = run_ollama(prompt)

    st.subheader("🧠 LLM Analysis")
    try:
        llm_json = json.loads(output)
        llm_verdict = llm_json.get("final_verdict", "Verdict not available")
        llm_confidence = llm_json.get("confidence", "N/A")
        st.subheader("✅ Final Verdict")
        st.write(f"**{llm_verdict}** (confidence: {llm_confidence})")
        st.json(llm_json)
    except Exception:
        st.write(output)

    with st.expander("🔍 Ranked Articles Used"):
        st.dataframe(pd.DataFrame(articles)[["source", "date", "rank_score", "text"]])
        
# ---------- RAG: Dummy Article Retrieval + LLM Reasoning ----------
#st.markdown("---")
#st.subheader("📚 RAG — Rank Dummy Articles Using Embeddings + LLM Analysis")

if st.button("Verify Claim through RAG + LLM"):
    if not claim.strip():
        st.warning("Please enter a claim first.")
        st.stop()

    # Dummy Articles
    articles = [
        {"source": "BBC", "date": "2024-11-02", "text": "Amazon announced it is expanding its data centers in Europe."},
        {"source": "CNN", "date": "2024-10-12", "text": "Amazon denies rumors about shutting down any European data centers."},
        {"source": "Reuters", "date": "2024-09-21", "text": "Tech companies continue to invest heavily in cloud infrastructure across Europe."},
        {"source": "TechCrunch", "date": "2024-11-11", "text": "Reports suggested Amazon restructured several European cloud regions but did not shut them down."},
        {"source": "NYTimes", "date": "2024-10-01", "text": "European governments welcome continued investment in technology expansion."},
        {"source": "WSJ", "date": "2024-08-18", "text": "Amazon evaluates performance and cost efficiency of its European cloud services."},
        {"source": "Forbes", "date": "2024-11-20", "text": "Cloud demand is growing rapidly across Europe leading to expansion of data facilities."},
        {"source": "Guardian", "date": "2024-09-09", "text": "Some speculated closures were misinformation spread on social platforms."},
        {"source": "Bloomberg", "date": "2024-10-30", "text": "Amazon confirms strategic upgrades to its European cloud infrastructure."},
        {"source": "Al Jazeera", "date": "2024-07-25", "text": "European tech landscape continues to see rapid cloud infrastructure growth."}
    ]

    with st.spinner("Generating embeddings and ranking articles..."):
        # Claim Embedding
        claim_emb = EMBED_MODEL.encode([claim])

        # Article Embeddings
        article_texts = [a["text"] for a in articles]
        article_embs = EMBED_MODEL.encode(article_texts)

        # Similarity
        sims = cosine_similarity(claim_emb, article_embs)[0]

        ranked = []
        for art, score, emb in zip(articles, sims, article_embs):
            ranked.append({
                "Source": art["source"],
                "Date": art["date"],
                "Text": art["text"],
                "Similarity Score": round(float(score), 4),
                "Embedding (preview)": emb[:10].tolist(),
                "Full Embedding": emb.tolist()
            })

        # Top 5 Ranked Articles
        ranked = sorted(ranked, key=lambda x: x["Similarity Score"], reverse=True)[:5]

    st.success("Top 5 Articles Retrieved & Ranked using Embeddings")

    st.subheader("🏆 Top 5 Ranked Articles")
    st.dataframe(pd.DataFrame(ranked)[["Source","Date","Text","Similarity Score","Embedding (preview)"]])

    with st.expander("🔍 View Full Embeddings of Each Article"):
        st.json(ranked)

    # --------------------------------------
    # FEED RANKED ARTICLES TO LLM
    # --------------------------------------
    st.subheader("🤖 LLM Reasoning on Retrieved Articles")

    top_articles_texts = [r["Text"] for r in ranked]

    with st.spinner("Running LLM analysis using RAG articles..."):
        prompt = build_prompt(claim, top_articles_texts)
        llm_output = run_ollama(prompt)

    st.subheader("🧠 LLM Analysis Result")

    try:
        llm_json = json.loads(llm_output)
        llm_verdict = llm_json.get("final_verdict", "Verdict not available")
        llm_confidence = llm_json.get("confidence", "N/A")

        st.subheader("✅ Final Verdict")
        st.write(f"**{llm_verdict}** (confidence: {llm_confidence})")

        st.json(llm_json)

    except Exception:
        st.error("LLM did not return valid JSON — raw output shown below")
        st.write(llm_output)


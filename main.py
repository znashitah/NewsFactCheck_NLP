from serpapi import GoogleSearch
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

def run_ollama(prompt):
    print(">>> Sending request to Ollama...")

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    print(">>> HTTP status:", response.status_code)
    print(">>> Raw response text:", response.text)

    response.raise_for_status()
    return response.json()["response"]


# ================= NLP FUNCTIONS =================
def sentence_similarity(claim, sentence):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([claim, sentence])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


def detect_stance(sentence):
    sentence = sentence.lower()
    if any(w in sentence for w in ["deny", "denied", "false", "not"]):
        return "Contradicts"
    if any(w in sentence for w in ["confirm", "confirmed", "announced", "expanding"]):
        return "Supports"
    return "Unverified"


def detect_sentiment(text):
    text = text.lower()
    pos = ["increase", "growth", "confirmed", "expanding"]
    neg = ["deny", "false", "shutting"]
    score = sum(w in text for w in pos) - sum(w in text for w in neg)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def ethical_warning(text):
    text = text.lower()
    if any(w in text for w in ["shocking", "must see", "breaking"]):
        return "Clickbait"
    return "None"

def build_prompt(claim, articles):
    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"""
Article {i+1}:
{article}
"""

    prompt = f"""
You are an expert fact-checking and claim verification AI.

User Claim:
"{claim}"

Below are raw statements extracted from online articles:
{articles_text}

Your tasks:
1. Determine the stance of EACH article toward the claim (support / refute / neutral)
2. Identify sentiment (positive / neutral / negative)
3. Detect misinformation or warning signals
4. Provide a confidence score (0–100)
5. Give a final verdict: True / False / Misleading / Unverifiable

Return ONLY valid JSON in the following format:

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
    return prompt
# ================= SERPAPI FETCH =================


def fetch_articles(claim, api_key, max_articles=5):
    print("\n>>> Fetching news using SerpApi")

    params = {
        "engine": "google_news",
        "q": claim,
        "hl": "en",
        "gl": "us",
        "num": max_articles,
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        #print(">>> SerpApi search object created")

        results = search.get_dict()
        #print(">>> SerpApi request completed")

    except Exception as e:
        print(">>> ERROR while fetching from SerpApi:", e)
        return []

    #print(">>> SerpApi response keys:", results)

    articles = []

    if "news_results" not in results:
        print(">>> No news_results found in response")
        return articles

    for item in results["news_results"]:
        #print(">>> Fetched:", item.get("title"))

        articles.append({
            "source": item.get("source", "Unknown"),
            "date": item.get("date", "N/A"),
            "text": (
                (item.get("title") or "") + ". " +
                (item.get("snippet") or "")
            )
        })

    return articles



def final_verdict(results):
    s = sum(r["Stance"] == "Supports" for r in results)
    c = sum(r["Stance"] == "Contradicts" for r in results)

    if s > c:
        return "Verified", round(s / len(results), 2)
    if c > s:
        return "Contradicted", round(c / len(results), 2)
    return "Unverified", 0.5


def print_timeline(results):
    print("\nNEWS TIMELINE")
    for r in sorted(results, key=lambda x: x["Date"]):
        print(f"{r['Date']} | {r['Source']} | {r['Stance']}")


# ================= MAIN =================
if __name__ == "__main__":

    claim = "Amazon is shutting down data centers in Europe"
    API_KEY = "8198373a9102fdb800c25e0c8337ff05cfce241afeb057f3d5a276588fee86dd"

    articles = fetch_articles(claim, API_KEY)

    if not articles:
        print("\n[WARNING] No articles found. Using fallback.")
        articles = [{
            "source": "BBC",
            "date": "2024-11-02",
            "text": "Amazon announced it is expanding its data centers in Europe."
        }]

    prompt = build_prompt(claim, articles)

    print("\n--- PROMPT SENT TO OLLAMA ---\n")
    print(prompt)

    print("\n--- OLLAMA ANALYSIS RESULT ---\n")
    result = run_ollama(prompt)
    print(result)
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

    # ================= OUTPUT TABLE =================
#    print("\n===== ANALYSIS RESULTS =====")
#    print(f"{'Source':<15}{'Date':<15}{'Rel':<6}{'Stance':<12}{'Sentiment':<10}{'Warnings':<12}Evidence")
#    print("-" * 120)

#for r in results:
    #print(
    #    f"{str(r['Source']):<15}{str(r['Date']):<15}{r['Relevance']:<6}"
    #    f"{r['Stance']:<12}{r['Sentiment']:<10}{r['Warnings']:<12}{r['Evidence']}"
    #)

    #verdict, confidence = final_verdict(results)
    #print("\nFINAL VERDICT:", verdict)
    #print("Confidence:", confidence)

    #print_timeline(results)

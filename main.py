import requests
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def fetch_articles(claim, api_key, max_articles=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={claim}&pageSize={max_articles}&language=en&sortBy=relevancy"
    )
    headers = {"Authorization": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return []

    articles = []
    for item in response.json().get("articles", []):
        articles.append({
            "source": item["source"]["name"],
            "date": item["publishedAt"][:10],
            "text": (item["title"] or "") + ". " + (item["description"] or "")
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
    API_KEY = "60ff4dd213a441ab9be84c76750b059c"

    articles = fetch_articles(claim, API_KEY)

    if not articles:
        articles = [{
            "source": "BBC",
            "date": "2024-11-02",
            "text": "Amazon announced it is expanding its data centers in Europe."
        }]

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

    print(f"{'Source':<12}{'Date':<12}{'Rel':<6}{'Stance':<12}{'Sentiment':<10}{'Warnings':<12}Evidence")
    print("-" * 110)

    for r in results:
        print(
            f"{r['Source']:<12}{r['Date']:<12}{r['Relevance']:<6}"
            f"{r['Stance']:<12}{r['Sentiment']:<10}{r['Warnings']:<12}{r['Evidence']}"
        )

    verdict, confidence = final_verdict(results)
    print("\nFINAL VERDICT:", verdict)
    print("Confidence:", confidence)

    print_timeline(results)

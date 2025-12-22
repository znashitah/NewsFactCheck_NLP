import requests

import nltk

# one-time downloads
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [w for w in tokens if w.isalpha() and w not in stop_words]


def sentence_similarity(claim, sentence):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([claim, sentence])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


def detect_stance(sentence):
    sentence = sentence.lower()
    contradict_words = ["deny", "denied", "no", "not", "false"]
    support_words = ["confirm", "confirmed", "announced", "is", "will"]

    for word in contradict_words:
        if word in sentence:
            return "Contradicts"

    for word in support_words:
        if word in sentence:
            return "Supports"

    return "Unverified"


def detect_sentiment(text):
    text = text.lower()
    positive_words = ["increase", "growth", "confirmed", "expanding", "investment"]
    negative_words = ["deny", "denied", "misinformation", "false", "shutting"]

    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


def final_verdict(results):
    support = sum(1 for r in results if r["Stance"] == "Supports")
    contradict = sum(1 for r in results if r["Stance"] == "Contradicts")
    total = len(results)

    if support > contradict:
        verdict = "Verified"
        confidence = support / total
    elif contradict > support:
        verdict = "Contradicted"
        confidence = contradict / total
    else:
        verdict = "Unverified"
        confidence = 0.5

    return verdict, round(confidence, 2)


def print_timeline(results):
    print("\nNEWS TIMELINE")
    print("-" * 50)
    sorted_results = sorted(results, key=lambda x: x["Date"])
    for r in sorted_results:
        print(f"{r['Date']} | {r['Source']} | {r['Stance']}")
def ethical_warning(text):
    text = text.lower()
    clickbait_words = ["shocking", "you won't believe", "amazing", "secret"]
    extreme_words = ["disaster", "catastrophic", "fake", "crisis"]
    manipulative_words = ["must see", "urgent", "breaking"]

    warnings = []

    if any(word in text for word in clickbait_words):
        warnings.append("Clickbait")
    if any(word in text for word in extreme_words):
        warnings.append("Extreme")
    if any(word in text for word in manipulative_words):
        warnings.append("Manipulative")

    if warnings:
        return ", ".join(warnings)
    else:
        return "None"

def fetch_articles(claim, api_key, max_articles=5):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={claim}&"
        f"pageSize={max_articles}&"
        f"language=en&"
        f"sortBy=relevancy"
    )

    headers = {"Authorization": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error fetching news:", response.status_code)
        return []

    data = response.json()
    articles = []

    for item in data.get("articles", []):
        articles.append({
            "source": item["source"]["name"],
            "date": item["publishedAt"][:10],  # YYYY-MM-DD
            "text": item["title"] + ". " + item.get("description", "")
        })
    return articles


# 🔹 FAKE NEWS ARTICLES
articles = [
    {
        "source": "BBC",
        "date": "2024-11-02",
        "text": "Amazon announced it is expanding its data centers in Europe. "
                "The company said demand for cloud services is increasing."
    },
    {
        "source": "Reuters",
        "date": "2024-11-01",
        "text": "Amazon denied reports of shutting down data centers. "
                "Executives confirmed continued investment in Europe."
    },
    {
        "source": "Al Jazeera",
        "date": "2024-11-03",
        "text": "There is no official confirmation that Amazon is closing data centers. "
                "Experts say misinformation is spreading online."
    }
]


if __name__ == "__main__":

    claim = "Amazon is closing data centers in Europe."
    print("CLAIM:", claim)
    print("=" * 100)

    results = []

    for article in articles:
        best_sentence = ""
        best_score = 0
        sentences = sent_tokenize(article["text"])

        for s in sentences:
            score = sentence_similarity(claim, s)
            if score > best_score:
                best_score = score
                best_sentence = s

        stance = detect_stance(best_sentence)
        sentiment = detect_sentiment(article["text"])
        warnings = ethical_warning(article["text"])

        results.append({
            "Source": article["source"],
            "Date": article["date"],
            "Relevance": round(best_score, 2),
            "Stance": stance,
            "Sentiment": sentiment,
            "Evidence": best_sentence,
            "Warnings": warnings
        })

    # ✅ Print verification table
    print(f"{'Source':<12} {'Date':<12} {'Rel':<6} {'Stance':<12} {'Sentiment':<10}{'Warnings':<15} Evidence")
    print("-" * 120)
    for r in results:
        print(
            f"{r['Source']:<12} "
            f"{r['Date']:<12} "
            f"{r['Relevance']:<6} "
            f"{r['Stance']:<12} "
            f"{r['Sentiment']:<10} "
            f"{r['Warnings']:<15} "
            f"{r['Evidence']}"
            
        )

    # ✅ Final verdict
    verdict, confidence = final_verdict(results)
    print("\nFINAL VERDICT")
    print("-" * 30)
    print("Verdict:", verdict)
    print("Confidence:", confidence)

    # ✅ Timeline
    print_timeline(results)

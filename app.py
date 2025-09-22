from flask import Flask, jsonify, render_template
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize Flask
app = Flask(__name__)

# Fetch top news from NewsAPI
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.json())
        return []
    return response.json().get("articles", [])

# Summarize text using HuggingFace Inference API
def summarize_text(text):
    if not text:
        return "No content available"

    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text, "parameters": {"max_length": 60, "min_length": 20}}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx
        result = response.json()

        # Check for errors returned by HF API
        if isinstance(result, dict) and result.get("error"):
            return f"Summarization failed: {result['error']}"

        # Normal response is a list of dicts with summary_text
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            return result[0]["summary_text"]

        return "Summarization failed: unexpected response format"

    except requests.exceptions.Timeout:
        return "Summarization failed: request timed out"
    except requests.exceptions.HTTPError as http_err:
        return f"Summarization failed: {http_err}"
    except Exception as e:
        return f"Summarization failed: {str(e)}"


# Simple sentiment analysis
from textblob import TextBlob
def get_sentiment(text):
    if not text:
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0: return "Positive"
    elif polarity < 0: return "Negative"
    else: return "Neutral"

# API endpoint
@app.route("/news", methods=["GET"])
def get_news():
    articles = fetch_news()
    summarized = []
    for a in articles[:10]:
        title = a.get("title", "No title")
        description = a.get("description", "")
        content = description or a.get("content", "")
        summary = summarize_text(content)
        sentiment = get_sentiment(content)
        summarized.append({
            "title": title,
            "summary": summary,
            "sentiment": sentiment,
            "source": a.get("source", {}).get("name", ""),
            "publishedAt": a.get("publishedAt", "")
        })
    return jsonify(summarized)

# Homepage
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

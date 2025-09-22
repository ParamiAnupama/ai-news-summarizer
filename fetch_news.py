import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

# Load API keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Hugging Face summarizer (runs locally, no API key needed)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 1. Fetch news from NewsAPI
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print("❌ Error fetching news:", data)
        return []

    return data.get("articles", [])

# 2. Summarize news content
def summarize_text(text):
    if not text:
        return "No content available"
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Summarization failed: {str(e)}"

# 3. Main script
if __name__ == "__main__":
    articles = fetch_news()
    print(f"✅ Found {len(articles)} articles\n")

    for idx, article in enumerate(articles[:5]):  # limit to 5 for demo
        title = article.get("title", "No title")
        description = article.get("description", "")
        content = description or article.get("content", "")

        summary = summarize_text(content)

        print(f"📰 {idx+1}. {title}")
        print(f"   ➡️ Summary: {summary}\n")

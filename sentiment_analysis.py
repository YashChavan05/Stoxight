import requests
import pandas as pd
from textblob import TextBlob
import yfinance as yf
from datetime import datetime, timedelta

def fetch_news_headlines(ticker, days_back=7):
    """
    Fetch news headlines for a given stock ticker.
    Args:
        ticker: Stock ticker symbol.
        days_back: Number of days to look back for news.
    Returns:
        list: List of news headlines.
    """
    try:
        # Using yfinance to get news
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        headlines = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for article in news:
            if 'title' in article:
                headlines.append(article['title'])
        
        return headlines[:10]  # Limit to 10 headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment(headlines):
    """
    Analyze sentiment of news headlines.
    Args:
        headlines: List of news headlines.
    Returns:
        dict: Sentiment analysis results.
    """
    if not headlines:
        return {'sentiment': 'neutral', 'score': 0, 'headlines': []}
    
    sentiments = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiments.append(blob.sentiment.polarity)
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    if avg_sentiment > 0.1:
        sentiment_label = 'positive'
    elif avg_sentiment < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        'sentiment': sentiment_label,
        'score': avg_sentiment,
        'headlines': headlines,
        'sentiments': sentiments
    }

def get_sentiment_analysis(ticker):
    """
    Get comprehensive sentiment analysis for a stock.
    Args:
        ticker: Stock ticker symbol.
    Returns:
        dict: Complete sentiment analysis.
    """
    headlines = fetch_news_headlines(ticker)
    sentiment_data = analyze_sentiment(headlines)
    
    return sentiment_data 
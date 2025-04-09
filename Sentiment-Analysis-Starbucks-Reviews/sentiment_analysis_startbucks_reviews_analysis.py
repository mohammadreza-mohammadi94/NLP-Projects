# -*- coding: utf-8 -*-
"""Sentiment Analysis of Starbucks Reviews

This script performs sentiment analysis on Starbucks reviews dataset from Kaggle.
It includes data preprocessing, exploratory data analysis (EDA), feature engineering,
and machine learning modeling with evaluation.

Original dataset: https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import re
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import string
import warnings
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt_tab")

# Load SpaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # Load once for efficiency

# --- Data Loading ---
def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

# --- Preprocessing ---
def clean_text(text: str) -> str:
    """Clean and preprocess text for sentiment analysis."""
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenization and lemmatization with SpaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text]
    # Remove stopwords and specific words
    stopwords = set(nltk.corpus.stopwords.words('english')) - {'not'}
    tokens = [token for token in tokens if token not in stopwords and token not in ['starbuck', 'starbucks', 'coffee']]
    return " ".join(tokens)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset."""
    # Drop unusable columns
    df = df.drop(columns=["name", "Image_Links"], errors='ignore')
    
    # Drop rows with NaN values
    df = df.dropna(axis=0)
    logger.info(f"Rows after dropping NaNs: {len(df)}")
    
    # Separate city and state from location
    df[['city', 'state']] = df['location'].str.rsplit(',', n=1, expand=True).apply(lambda x: x.str.strip())
    df = df.drop('location', axis=1)
    
    # Convert Date column to datetime
    df['Date'] = df['Date'].str.replace('Reviewed', '').str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Extract year and month
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
    # Filter out invalid states
    states_to_remove = [None, "Other", "OTHER", "Saskatchewan", "NO OTHER LINE NEEDED", "UK", "Alberta"]
    df = df[~df['state'].isin(states_to_remove)]
    
    # Classify sentiment based on rating
    df['Sentiment'] = df['Rating'].apply(lambda x: "Satisfied" if x >= 3 else "Unsatisfied")
    
    # Apply text cleaning
    df['preprocessed_review'] = df['Review'].apply(clean_text)
    logger.info("Text preprocessing completed")
    
    return df

# --- Exploratory Data Analysis (EDA) ---
def plot_sentiment_pie(df: pd.DataFrame) -> None:
    """Plot a pie chart of sentiment distribution."""
    plt.figure(figsize=(8, 5))
    plt.pie(df['Sentiment'].value_counts(), labels=df['Sentiment'].value_counts().index,
            autopct='%1.1f%%', colors=sns.color_palette("Set1"))
    plt.title('Sentiment Distribution (Satisfied vs Unsatisfied)', fontsize=15, fontweight='bold')
    plt.savefig('sentiment_pie.png')
    plt.show()

def plot_rating_pie(df: pd.DataFrame) -> None:
    """Plot a pie chart of rating distribution."""
    plt.figure(figsize=(8, 5))
    plt.pie(df['Rating'].value_counts(), labels=df['Rating'].value_counts().index,
            autopct="%1.1f%%", colors=sns.color_palette("Set1"))
    plt.title('Rating Distribution', fontsize=15, fontweight='bold')
    plt.savefig('rating_pie.png')
    plt.show()

def plot_rating_trend(df: pd.DataFrame) -> None:
    """Plot a line chart of rating trends over years."""
    df_grp = df.groupby(['year', 'Sentiment'], as_index=False)['Rating'].count()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_grp, x="year", y="Rating", hue="Sentiment", lw=3, palette="Set1")
    plt.title("Rating Trend Over Years", fontsize=15, fontweight='bold')
    plt.xlim([2012, 2023])
    plt.xticks(range(2012, 2024))
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.ylim([0, df_grp['Rating'].max() + 10])
    plt.savefig('rating_trend.png')
    plt.show()

def plot_wordcloud(df: pd.DataFrame, sentiment: str) -> None:
    """Generate and plot a word cloud for a specific sentiment."""
    text = " ".join(df[df['Sentiment'] == sentiment]['preprocessed_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment} Reviews", fontsize=15, fontweight='bold')
    plt.savefig(f'wordcloud_{sentiment.lower()}.png')
    plt.show()

# --- Feature Engineering & Modeling ---
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional features from reviews."""
    X = df[['Review', 'preprocessed_review']].copy()
    X["char_count"] = X['Review'].apply(lambda x: len(x.replace(" ", "")))
    X["word_count"] = X['Review'].apply(lambda x: len(x.split()))
    X["word_density"] = X["char_count"] / (X["word_count"] + 1)
    X["punctuation_count"] = X["Review"].apply(lambda x: len("".join(i for i in x if i in string.punctuation)))
    return X

def train_model(X_train_tfidf, X_test_tfidf, y_train, y_test) -> None:
    """Train and evaluate a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    logger.info("Model Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unsatisfied', 'Satisfied']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unsatisfied', 'Satisfied'],
                yticklabels=['Unsatisfied', 'Satisfied'])
    plt.title("Confusion Matrix", fontsize=15, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix.png')
    plt.show()

# --- Main Execution ---
def main():
    """Main function to run the sentiment analysis pipeline."""
    # Load dataset
    df = load_data("reviews_data.csv")
    if df.empty:
        return
    
    # Preprocess data
    df = preprocess_data(df)
    df.to_csv("preprocessed_reviews.csv", index=False)
    logger.info("Preprocessed data saved to 'preprocessed_reviews.csv'")
    
    # EDA
    plot_sentiment_pie(df)
    plot_rating_pie(df)
    plot_rating_trend(df)
    plot_wordcloud(df, "Satisfied")
    plot_wordcloud(df, "Unsatisfied")
    
    # Feature engineering
    X = extract_features(df)
    y = df['Sentiment'].map({'Satisfied': 1, 'Unsatisfied': 0})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorization
    tfidf_vec = TfidfVectorizer(analyzer='word', max_features=4000)
    X_train_tfidf = tfidf_vec.fit_transform(X_train['preprocessed_review'])
    X_test_tfidf = tfidf_vec.transform(X_test['preprocessed_review'])
    
    logger.info(f"TF-IDF Vectorization completed: {X_train_tfidf.shape[1]} features")
    
    # Train and evaluate model
    train_model(X_train_tfidf, X_test_tfidf, y_train, y_test)

if __name__ == "__main__":
    # Ensure SpaCy model is downloaded
    try:
        spacy.load('en_core_web_sm')
    except OSError:
        logger.info("Downloading SpaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
    
    main()
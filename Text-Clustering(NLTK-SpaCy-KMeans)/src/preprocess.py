import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Download NLTK resources
def setup_nltk():
    """Download required NLTK resources."""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        nltk.download("movie_reviews")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")

# Preprocess text function
class TextPreprocessor:
    """Preprocess text data for NLP tasks."""
    def __init__(self, lang="en_core_web_lg"):
        try:
            self.nlp = spacy.load(lang, disable=["parser", "ner"])
            self.stop_words = set(stopwords.words('english'))
            self.punctuation = set(string.punctuation)
            logger.info(f"Initialized TextPreprocessor: {lang}")
        except Exception as e:
            logger.error(f"Failed to initialize preprocessor: {e}")
            raise e
        
    def preprocess(self, texts):
        """Preprocess texts by tokenizing, lemmatizing, and removing stopwords/punctuation."""
        try:
            processed_texts = []
            for doc in self.nlp.pipe(texts, batch_size=100):
                tokens = [
                    token.lemma_.lower() for token in doc
                    if token.text.lower() not in self.stop_words
                    and token.text not in self.punctuation
                    and token.is_alpha
                ]
                processed_texts.append(tokens)
            logger.info(f"Preprocessed {len(processed_texts)} texts")
            return processed_texts
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise e
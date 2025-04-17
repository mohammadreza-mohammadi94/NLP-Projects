import numpy as np
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import spacy
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from preprocessed texts for clustering."""
    def __init__(self, vector_size=100, window=5, min_count=5, sg=1):
        """Initialize the feature extraction parameters."""
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.w2v_model = None
        self.dictionary = None
        self.tfidf_model = None
        self.nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
        logger.info("FeatureExtraction initialized with vector_size=%d, window=%d, min_count=%d, sg=%d",
                    vector_size, window, min_count, sg)

    def train_word2vec(self, tokens):
        """Train Word2Vec model on tokenized texts."""
        try:
            self.w2v_model = Word2Vec(
                tokens,
                vector_size= self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                epochs=10,
                workers=4
            )
            logger.info("Word2Vec Model Trained")
        except Exception as e:
            logger.error(f"Error training Word2Vec model: {e}")
            raise e

    def get_word2vec_features(self, tokens):
        """Generate document vectors using Word2Vec."""
        try:
            if not self.w2v_model:
                self.train_word2vec(tokens)
            X = np.array([
                np.mean([self.w2v_model.wv[word] for word in doc if word in self.w2v_model.wv], axis=0)
                if any(word in self.w2v_model.wv for word in doc)
                else np.zeros(self.vector_size)
                for doc in tokens
            ])
            logger.info(f"Generated Word2Vec features: {X.shape}")
            return X
        except Exception as e:
            logger.error(f"Word2Vec feature extraction error: {e}")
            raise e

    def get_tfidf_features(self, tokens):
        """Generate TF-IDF features."""
        try:
            self.dictionary = Dictionary(tokens)
            self.dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [self.dictionary.doc2bow(doc) for doc in tokens]
            self.tfidf_model = TfidfModel(corpus)
            X = np.zeros((len(corpus), len(self.dictionary)))
            for doc_idx, doc in enumerate(corpus):
                tfidf_doc = self.tfidf_model[doc]
                for term_idx, tfidf_value in tfidf_doc:
                    X[doc_idx, term_idx] = tfidf_value
            logger.info(f"Generated TF-IDF features: {X.shape}")
            return X
        except Exception as e:
            logger.error(f"TF-IDF feature extraction failed: {e}")
            raise e
            
    def get_spacy_features(self, texts):
        """Generate document vectors using SpaCy."""
        try:
            docs = list(self.nlp.pipe(texts, batch_size=100))
            X = np.array([doc.vector for doc in docs])
            logger.info(f"Generated SpaCy features with shape: {X.shape}")
            return X
        except Exception as e:
            logger.info(f"Spacy feature extraction failed: {e}")
            raise e
        

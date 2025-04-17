import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from nltk.corpus import movie_reviews
import nltk
from src.preprocess import TextPreprocessor, setup_nltk
from src.feature_extraction import FeatureExtractor
from src.clustering import TextClusterer
from src.visualization import visualize_clusters, generate_cluster_report
import joblib

# Create output directories before logging
Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output/word2vec_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data() -> list:
    """Load movie reviews data."""
    try:
        setup_nltk()
        texts = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
        logger.info(f"Loaded {len(texts)} reviews")
        return texts
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise e

def main():
    """Main function to execute the text clustering pipeline."""
    try:
        # Load data
        texts = load_data()

        # Preprocess texts
        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess(texts)
        with open("data/preprocessed_tokens.pkl", "wb") as f:
            pickle.dump(tokens, f)

        # Extract features
        extractor = FeatureExtractor()
        X_w2v = extractor.get_word2vec_features(tokens)
        X_tfidf = extractor.get_tfidf_features(tokens)
        X_spacy = extractor.get_spacy_features(texts)

        # Save Word2Vec model
        extractor.w2v_model.save("models/word2vec.model")

        # Cluster texts
        clusterer = TextClusterer(n_clusters=5, num_topics=5)
        
        # Optimize K for K-Means
        optimal_k = clusterer.optimize_k(X_w2v)
        clusterer.n_clusters = optimal_k
        
        # Apply K-Means
        clusters_kmeans, silhouette_score = clusterer.apply_kmeans(X_w2v)
        joblib.dump(clusterer.kmeans, "models/kmeans_model.pkl")

        # Apply LDA
        corpus = [extractor.dictionary.doc2bow(doc) for doc in tokens]
        clusters_lda, lda_model = clusterer.apply_lda(corpus, extractor.dictionary)
        lda_model.save("models/lda_model")

        # Generate reports
        report_kmeans = generate_cluster_report(clusters_kmeans, tokens, "output/kmeans_cluster_report.csv")
        report_lda = generate_cluster_report(clusters_lda, tokens, "output/lda_cluster_report.csv")
        
        # Visualize clusters
        visualize_clusters(X_w2v, clusters_kmeans, "output/kmeans_clusters_tsne.png")
        visualize_clusters(X_w2v, clusters_lda, "output/lda_clusters_tsne.png")

        # Log final results
        logger.info(f"K-Means Report:\n{report_kmeans.to_string()}")
        logger.info(f"LDA Report:\n{report_lda.to_string()}")
        logger.info(f"Final Silhouette Score (K-Means): {silhouette_score:.3f}")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise e

if __name__ == "__main__":
    main()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import LdaModel
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class TextClusterer:
    """Perform text clustering using K-Means and LDA."""
    
    def __init__(self, n_clusters = 5, num_topics = 5):
        """Initialize with clustering parameters."""
        self.n_clusters = n_clusters
        self.num_topics = num_topics
        self.kmeans = None
        self.lda = None
        logger.info(f"Initialized TextClusterer with {n_clusters} clusters and {num_topics} topics")

    def apply_kmeans(self, X):
        """Apply K-Means clustering."""
        try:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = self.kmeans.fit_predict(X)
            score = silhouette_score(X, clusters)
            logger.info(f"K-Means clustering completed. Silhouette Score: {score:.3f}")
            return clusters, score
        except Exception as e:
            logger.error(f"K-Means clustering error: {e}")
            raise e

    def apply_lda(self, corpus, dictionary):
        """Apply LDA for topic modeling."""
        try:
            self.lda = LdaModel(
                corpus,
                num_topics=self.num_topics,
                id2word=dictionary,
                passes=10,
                random_state=42
            )
            clusters = np.array([max(self.lda[doc], key=lambda x: x[1])[0] for doc in corpus])
            logger.info("LDA clustering completed")
            return clusters, self.lda
        except Exception as e:
            logger.error(f"LDA clustering error: {e}")
            raise e

    def optimize_k(self, X, k_range = range(2, 11)):
        """Find optimal number of clusters using Silhouette Score."""
        try:
            best_k, best_score = 2, -1
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X)
                score = silhouette_score(X, clusters)
                logger.info(f"K={k}, Silhouette Score={score:.3f}")
                if score > best_score:
                    best_k, best_score = k, score
            logger.info(f"Optimal K: {best_k} with Silhouette Score: {best_score:.3f}")
            self.n_clusters = best_k
            return best_k
        except Exception as e:
            logger.error(f"K optimization error: {e}")
            raise e
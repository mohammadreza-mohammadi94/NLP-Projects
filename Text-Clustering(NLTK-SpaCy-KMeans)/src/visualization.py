import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def visualize_clusters(X, clusters, output_path = "clusters_tsne.png"):
    """Visualize clusters using t-SNE."""
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_2d = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            x=X_2d[:, 0], y=X_2d[:, 1],
            hue=clusters, palette="viridis", size=clusters, sizes=(50, 200), alpha=0.7
        )
        plt.title("Text Clusters Visualized with t-SNE", fontsize=14, fontweight="bold")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Cluster")
        plt.savefig(output_path, dpi=300)
        plt.show()
        logger.info(f"Visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise e

def generate_cluster_report(clusters, tokens, output_path = "cluster_report.csv"):
    """Generate a report with cluster details."""
    try:
        from collections import defaultdict
        cluster_docs = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_docs[cluster].append(tokens[i])
        
        report = []
        for cluster_id, docs in cluster_docs.items():
            word_counts = Counter(word for doc in docs for word in doc)
            top_words = word_counts.most_common(5)
            report.append({
                "Cluster": cluster_id,
                "Size": len(docs),
                "Top Words": ", ".join([f"{w} ({c})" for w, c in top_words])
            })
        
        df = pd.DataFrame(report)
        df.to_csv(output_path, index=False)
        logger.info(f"Cluster report saved to {output_path}")
        return df
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise e
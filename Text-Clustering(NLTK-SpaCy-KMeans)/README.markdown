# Text Clustering Project

This project performs text clustering on movie reviews to identify semantic groups (e.g., positive/negative sentiments, thematic topics). It uses NLTK, SpaCy, and Gensim for preprocessing, feature extraction, and clustering, with K-Means and LDA algorithms.

## Installation

1. Create and activate a Conda environment:

   ```bash
   conda create -n clustering_project python=3.10
   conda activate clustering_project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

3. Download NLTK data:

   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   nltk.download("wordnet")
   nltk.download("movie_reviews")
   ```

## Usage

Run the main script:

```bash
python main.py
```

## Outputs

- `output/clusters_tsne.png`: t-SNE visualization of clusters.
- `output/cluster_report.csv`: Cluster details (size, top words).
- `models/`: Saved Word2Vec, K-Means, and LDA models.
- `data/preprocessed_tokens.pkl`: Preprocessed data.

## Structure

- `src/preprocess.py`: Text preprocessing (tokenization, lemmatization).
- `src/feature_extraction.py`: Feature extraction (TF-IDF, Word2Vec).
- `src/clustering.py`: Clustering with K-Means and LDA.
- `src/visualization.py`: Visualization with t-SNE and reports.
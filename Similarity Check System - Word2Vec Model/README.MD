# Word2Vec Similarity System

This project implements a Word2Vec-based system to analyze word similarity using the Reuters news corpus. It includes training a Word2Vec model, retrieving similar words, visualizing word embeddings in 2D space with t-SNE, and generating analytical reports.

## Features

- ✅ Load and preprocess the **Reuters corpus** from NLTK  
- ✅ Train a custom **Word2Vec model** using Gensim  
- ✅ Find **top-N similar words** for any given input word  
- ✅ Visualize word embeddings using **t-SNE**  
- ✅ Generate similarity reports and corpus statistics in **CSV format**  
- ✅ Integrated logging and error handling for robust execution

---

## Installation

Run the following command to install required dependencies:

```bash
pip install -q gensim nltk matplotlib seaborn scikit-learn pandas
```

---

## Usage

### 1. Run the Script

```bash
python your_script_name.py
```

> By default, the system processes 1000 documents from the Reuters corpus and uses Skip-Gram Word2Vec architecture.

### 2. Output

- 📄 `output/similarity_report.csv` – list of similar words for each test input  
- 📄 `output/corpus_stats.csv` – total/unique word count and most common terms  
- 🖼️ `output/word_embeddings_tsne.png` – 2D scatter plot of word embeddings

---

## Configuration

You can customize model parameters like this (inside the script):

```python
system = Word2VecSimilaritySystem(
    max_docs=1000,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,          # 1 = Skip-gram, 0 = CBOW
    epochs=10
)
```

---

## Sample Test Words

The following words are used for similarity search and visualization:

```python
["market", "stock", "trade", "economy", "bank"]
```

---

## Project Structure

```
├── Word2VecSimilaritySystem.py   # Main script
├── output/
│   ├── similarity_report.csv     # Word similarity results
│   ├── corpus_stats.csv          # Vocabulary statistics
│   └── word_embeddings_tsne.png  # Embedding visualization
└── word2vec.log                  # Log file with runtime info
```

---

## License

This project is open-source and free to use for educational and research purposes.

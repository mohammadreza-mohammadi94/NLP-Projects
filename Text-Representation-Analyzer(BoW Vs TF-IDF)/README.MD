# 📊 Text Representation Analyzer (TF-IDF & BoW)

This project explores **text representation techniques** using **TF-IDF** and **Bag of Words (BoW)** on a small text corpus.

It helps you:
- Apply different configurations of TF-IDF and BoW
- Compare their sparsity and vocabulary sizes
- Display and save results for further analysis

---

## 🚀 Features

- ✅ TF-IDF with:
  - Stopword removal
  - N-gram support (unigrams & bigrams)
- ✅ Simple Count-based BoW
- 📊 Visual comparison (sparsity & vocab size)
- 📁 Save all results as a CSV file

---

## 📦 Dependencies

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ Usage

```python
from your_script import TextRepresentationAnalyzer

corpus = [
    "The cat sleeps quietly in the room",
    "The dog runs fast outside",
    ...
]

analyzer = TextRepresentationAnalyzer(corpus)
analyzer.run_experiments()
analyzer.display_results()
analyzer.compare_sparsity_and_vocab_size()
analyzer.save_results("tfidf_bow_results.csv")
```

---

## 📁 Output

- Printed matrices for each method
- Sparsity and vocabulary size plots (`comparison_plot.png`)
- CSV file with all results (`tfidf_bow_results.csv`)

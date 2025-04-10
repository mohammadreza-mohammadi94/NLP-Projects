# 🧠 BoWExplorer

**BoWExplorer** is a Python class for exploring different variations of the Bag-of-Words (BoW) technique on a given corpus. It helps you apply, visualize, and compare different BoW representations, including binary, n-grams, and stopword filtering.

---

## 🚀 Features

- 📦 Apply multiple BoW strategies:
  - Count BoW
  - Binary BoW
  - N-grams (e.g., unigrams + bigrams)
  - Stopword removal  
- 📊 Display BoW matrices as clean Pandas DataFrames  
- 🔍 Compare sparsity across methods (with visualization)  
- 💾 Export all results to a CSV file  

---

## 📂 Example Usage

```python
corpus = [
    "Natural language processing is fun.",
    "Bag of Words is a simple NLP method.",
    "NLP includes tokenization and lemmatization."
]

explorer = BoWExplorer(corpus)
explorer.run_experiments()
explorer.display_results()
explorer.compare_sparsity()
explorer.save_results("bow_results.csv")
```

---

## 📋 Requirements

- Python 3.7+  
- `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `numpy`

Install with:

```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

## 📧 Contact

**Author:** Mohammad Reza Mohammadi  
📩 Email: mr.mhmdi93@gmail.com

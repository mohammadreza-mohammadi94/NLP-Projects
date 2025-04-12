# 🧠 SpaCy Keyphrase Extraction with Machine Learning

This project implements a custom **keyphrase extraction system** built on top of [SpaCy](https://spacy.io/) and [NLTK's Reuters Corpus](https://www.nltk.org/nltk_data/). It combines **linguistic rules** with **machine learning (Logistic Regression)** to identify and rank important keyphrases from news articles.

---

## 📌 Features

- ✅ Custom SpaCy pipeline for linguistic pattern extraction
- ✅ Advanced rule-based phrase matching (NOUN+NOUN, ADJ+NOUN, subject+verb)
- ✅ Named Entity Recognition (NER) integration
- ✅ Feature engineering: phrase length, frequency, TF-IDF, entity presence
- ✅ Logistic Regression-based scoring of phrases
- ✅ Visual report: score distribution plots
- ✅ Auto-generated CSV reports

---

## 📁 Project Structure

```
.
├── KeyphraseExtractor.ipynb        # Main Jupyter notebook
├── output/
│   ├── keyphrase_results.csv       # Output CSV file with phrases and scores
│   └── score_distribution.png      # Score histogram
└── README.md                       # Project documentation
```

---

## ⚙️ How It Works

### 1. Load and Preprocess Data
- Loads documents from the Reuters Corpus (first 10 by default).
- Uses `SpaCy` for tokenization, POS tagging, and dependency parsing.

### 2. Keyphrase Extraction
- Adds a **custom SpaCy component** to extract phrases via:
  - Noun chunk patterns (`NOUN+NOUN`, `ADJ+NOUN`)
  - Dependency patterns (`nsubj + VERB`, `dobj + VERB`)
  - Named Entities

### 3. Feature Computation
Each phrase is scored using:
- **Length**: number of words in the phrase
- **Frequency**: count across the corpus
- **TF-IDF**: uniqueness measure
- **NER Flag**: if it's a named entity

### 4. Machine Learning Model
- A Logistic Regression model is trained on **synthetic labeled data**
- It predicts an **importance score** (probability) for each phrase

### 5. Visualization & Reporting
- Outputs a **histogram of scores**
- Prints ranked keyphrases per document
- Saves results to CSV

---

## 📊 Example Output

```
Text 1: The dollar ended mixed in quiet European trading Tuesday ...
Keyphrases:
- economic growth (Score: 0.902)
- trade agreement (Score: 0.873)
- U.S. (Score: 0.769)

Summary:
- Total Keyphrases: 120
- Average Score: 0.563
- Unique Keyphrases: 78
```

---

## 🛠 Installation & Setup

### 📦 Requirements
- Python 3.7+
- SpaCy
- NLTK
- scikit-learn
- pandas, matplotlib, seaborn

### 🔧 Install Dependencies

```bash
pip install spacy nltk scikit-learn pandas matplotlib seaborn
python -m nltk.downloader reuters
python -m spacy download en_core_web_sm
```

---

## 🚀 Run the Project

```bash
python KeyphraseExtractor.ipynb
```

Or use in **interactive mode** with Jupyter Notebook or Google Colab.

---

## 📈 Output Files

| File | Description |
|------|-------------|
| `output/keyphrase_results.csv` | Ranked keyphrases with features and scores |
| `output/score_distribution.png` | Histogram of phrase importance |

---

## 🤖 ML Model Info

The Logistic Regression model is trained on manually crafted synthetic samples simulating useful and non-useful keyphrases. While not perfect, it provides a decent baseline for ranking based on heuristic features.

---

## 🧪 Future Improvements

- Replace synthetic training data with annotated ground truth
- Incorporate deep learning models (e.g., BERT-based extractors)
- Extend phrase rules for more linguistic coverage
- Build a Streamlit or Flask web app for interactive demo

---

## 📚 References

- [SpaCy Documentation](https://spacy.io/usage)
- [NLTK Reuters Corpus](https://www.nltk.org/nltk_data/)
- [scikit-learn](https://scikit-learn.org/)

---

## 🧑‍💻 Author

**Mohammad Reza Mohammadi**  
[GitHub](https://github.com/mohammadreza-mohammadi94)

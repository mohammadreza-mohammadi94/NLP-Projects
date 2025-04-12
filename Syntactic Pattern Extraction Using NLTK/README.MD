# 📊 NLTK Custom Syntactic Pattern Extraction

This project performs **custom syntactic pattern extraction** from the Brown Corpus using **NLTK**, with a focus on specific part-of-speech (POS) tag sequences (e.g., `NOUN+VERB`, `ADJ+NOUN`). It includes pattern analysis, economic-topic filtering, and visualizations.

📁 [Original Colab Notebook](https://colab.research.google.com/drive/1A3AhOlVEcRFEbByaguOU-MBe2_EQJ-Zs)

---

## ✨ Features

- ✅ POS tagging with simplified categories (NOUN, VERB, ADJ, ADV)
- ✅ Extraction of user-defined syntactic patterns
- ✅ Sample phrase collection per pattern
- ✅ Economic-topic sentence detection
- ✅ Statistical analysis of patterns
- ✅ Comparison of overall vs. economic pattern usage
- ✅ Beautiful visualizations with Seaborn
- ✅ Export to CSV

---

## 🧠 Targeted Patterns

This tool focuses on the following POS tag combinations:
- `NOUN + VERB` (e.g., *jury said*)
- `ADJ + NOUN` (e.g., *grand jury*)
- `VERB + ADV` (e.g., *said quickly*)
- `NOUN + NOUN` (e.g., *stock market*)
- `VERB + NOUN` (e.g., *chase cats*)

---

## 🧰 Dependencies

Install the required libraries:

```bash
pip install nltk pandas matplotlib seaborn
```

Download necessary NLTK resources:

```python
import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## 🚀 Usage

Run the script directly from the command line:

```bash
python syntactic_pattern_extractor.py
```

The script will:

1. Load the **Brown Corpus** (`news` category by default).
2. Tokenize and POS-tag sentences.
3. Extract and count syntactic patterns.
4. Identify economic-related patterns (optional).
5. Analyze frequency and relevance.
6. Visualize and save results.

---

## 📊 Output

- 📄 `output/pattern_analysis.csv` — Table of pattern statistics
- 📈 `output/pattern_distribution.png` — Bar plot of pattern frequencies
- 📉 `output/economic_comparison.png` — Economic vs. overall pattern comparison

---

## 🧪 Sample Output

**Top Patterns**
| Pattern | Count | Percentage (%) | Sample Matches |
|--------|-------|----------------|----------------|
| ADJ+NOUN | 112 | 15.2 | grand jury, economic growth, ... |

**Economic Patterns**
| Pattern | Economic Count | Economic Percentage (%) |
|--------|----------------|--------------------------|
| NOUN+NOUN | 45 | 38.2 |

---

## 📂 Project Structure

```
.
├── syntactic_pattern_extractor.py  # Main script
├── output/
│   ├── pattern_analysis.csv
│   ├── pattern_distribution.png
│   └── economic_comparison.png
└── README.md
```

---

## 🔍 Example Use Case

This tool can be used in:
- Linguistic research
- Corpus analysis
- NLP pattern mining
- Economic sentiment detection in text

---

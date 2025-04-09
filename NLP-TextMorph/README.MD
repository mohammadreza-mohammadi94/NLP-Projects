# 🧠 NLP Text Morph

This repository provides a clean and modular implementation of common NLP preprocessing steps using **NLTK** in Python. It includes tokenization, POS tagging, lemmatization (with and without POS), and stopword removal—all wrapped inside a reusable `TextProcessor` class.

## 🚀 Features

- Tokenizes input text  
- Removes English stopwords  
- Applies POS tagging  
- Lemmatizes words using and without POS tags  
- Logs each processing step for better traceability  
- Displays results as a formatted DataFrame

## 📦 Dependencies

- Python 3.7+
- NLTK
- pandas

## ▶️ Quick Start

```bash
python nlp_text_morph.py
```

## 📝 Sample Output

```
 Original Tokens     POS Tags Lemmatized with POS Lemmatized without POS
           cats             NNS               cat                  cat
         running           VBG             run                running
         faster             RBR             fast                 faster
         jumping           VBG             jump              jumping
          better            RBR             well                better
         gardens           NNS             garden             garden
```

## 📁 Structure

- `TextProcessor`: Encapsulates all NLP steps
- `main()`: Demo pipeline on sample text

## 💡 Ideal For

- NLP preprocessing tutorials  
- Feature engineering in ML pipelines  
- Quick prototyping for text-based projects  

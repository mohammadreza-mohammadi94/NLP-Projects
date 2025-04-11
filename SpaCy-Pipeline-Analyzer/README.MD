# SpaCy Pipeline Analyzer

This project demonstrates how to explore and extend SpaCy's NLP pipeline with both built-in components and a custom token-level attribute.

## 🧠 What It Does

- Loads a SpaCy model (`en_core_web_sm`)
- Processes a small sample text corpus
- Extracts key linguistic features:
  - POS tags
  - Lemmas
  - Dependency labels
- Adds a **custom pipeline component** to flag words starting with the letter `'c'`
- Displays results in tabular format
- Visualizes:
  - POS tag distribution
  - Custom attribute (`starts_with_c`) distribution
- Saves all results to a CSV file

## 📁 File Structure

- `SpaCy-Pipelines-Analyzer.ipynb`: Main notebook with all code
- `spacy_pipeline_results.csv`: Output CSV with processed token data
- `pipeline_distribution.png`: Visualization of POS tags and custom attribute

## 🚀 How to Use

1. Install required packages:
    ```bash
    pip install spacy pandas matplotlib seaborn
    python -m spacy download en_core_web_sm
    ```

2. Run the notebook or script:
    ```bash
    python SpaCy-Pipelines-Analyzer.py
    ```

3. Check:
    - Console for tabular outputs
    - Saved CSV file (`spacy_pipeline_results.csv`)
    - Plot image (`pipeline_distribution.png`)

## ✨ Features

- Built-in logging
- Custom token extensions in SpaCy
- Clean and modular class-based design
- Visual and tabular insights

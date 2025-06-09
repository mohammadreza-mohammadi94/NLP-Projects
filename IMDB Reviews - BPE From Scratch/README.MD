
# Byte Pair Encoding (BPE) From Scratch

This project implements Byte-Pair Encoding (BPE) from scratch using Python. It demonstrates how to tokenize a text corpus into subword units, a common technique used in modern NLP models like GPT and BERT.

## Features

* Loads and preprocesses the IMDB movie reviews dataset
* Implements core BPE functions:

  * Pair frequency counting
  * Merge operations
  * Tokenization using learned BPE rules
* Compares BPE tokenization with NLTK word tokenization
* Saves learned merge rules to a text file

## Dependencies

Install required packages:

```bash
pip install pandas nltk kagglehub
```

Also, make sure to download NLTK resources:

```python
import nltk
nltk.download("punkt")
```

## Dataset

The dataset is downloaded from [KaggleHub](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

Make sure to authenticate KaggleHub and access the file path correctly for your environment.

## How It Works

1. **Preprocess Text:** Lowercase, remove punctuation, and split into words
2. **Build Corpus:** From IMDB reviews
3. **Learn BPE:** Merge most frequent symbol pairs iteratively
4. **Tokenize Reviews:** Using the learned BPE merges
5. **Compare:** Token counts and output with NLTK tokenization

## Output

* Learned vocabulary and BPE rules
* Tokenized reviews (subword level)
* Token count analysis
* Rule file: `bpe_merges.txt`

## Example

```python
apply_bpe("interesting", merges)
# Output: ['in', 'ter', 'est', 'ing</w>']
```

## License

This project is for educational purposes.


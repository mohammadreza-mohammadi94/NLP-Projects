# -*- coding: utf-8 -*-
"""Text-Summarization-NLP-ExtractiveMethod.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FaD07Q-tLDHlgCAAj2tdo2XrDIOSuWSG

# Word Frequency Based
"""

import re
import heapq
import nltk
import spacy
from nltk.corpus import stopwords
from collections import defaultdict

# NLTK
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download("wordnet")

text = """
A litany of text summarization methods have been developed over the last several decades, so answering how text summarization works doesn’t have a single answer.
This having been said, these methods can be classified according to their general approaches in addressing the challenge of text summarization.
Perhaps the most clear-cut and helpful distinction is that between Extractive and Abstractive text summarization methods. Extractive methods seek to extract the most pertinent information from a text.
Extractive text summarization is the more traditional of the two methods, in part because of their relative simplicity compared to abstractive methods.
Abstractive methods instead seek to generate a novel body of text that accurately summarizes the original text. Already we can see how this is a more difficult problem - there is a significant degree of freedom in not being limited to simply returning a subset of the original text. This difficulty comes with an upside, though.
Despite their relative complexity, Abstractive methods produce much more flexible and arguably faithful summaries, especially in the age of Large Language Models.
As mentioned above, Extractive Text Summarization methods work by identifying and extracting the salient information in a text.
The variety of Extractive methods therefore constitutes different ways of determining what information is important (and therefore should be extracted).
For example frequency-based methods will tend to rank the sentences in a text in order of importance by how frequently different words are used.
For each sentence, there exists a weighting term for each word in the vocabulary, where the weight is usually a function of the importance of the word itself and the frequency with which the word appears throughout the document as a whole.
Using these weights, the importance of each sentence can then be determined and returned.
Graph-based methods cast textual documents in the language of mathematical graphs.
In this schema, each sentence is represented as a node, where nodes are connected if the sentences are deemed to be similar.
What constitutes “similar” is, again, a choice of different specific algorithms and approaches.
For example, one implementation might use a threshold on the cosine similarity between TF-IDF vectors. In general, the sentences that are globally the “most similar” to all other sentences (i.e. those with the highest centrality) in the document are considered to have the most summarizing information,
and are therefore extracted and put into the summary.
A notable example of a graph-based method is TextRank, a version of Google’s pagerank algorithm (which determines what results to display in Google Search) that has been adapted for summarization (instead ranking the most important sentences).
Graph-based methods may benefit in the future from advances in Graph Neural Networks.
"""

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text) # Removes whitespaces
    text = re.sub(r"\W", " ", text) # Removes non alphabetic char
    text = re.sub(r"\d+", "", text) # Removes digits
    words = nltk.word_tokenize(text) # Tokenization
    words = [word for word in words if word not in stopwords.words("english")] # Remove stopwords
    words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words] # Lemmatization
    return words

clean_text = preprocess_text(text)

# Words frequency analysis
word_freq = defaultdict(int)
for word in clean_text:
    word_freq[word] += 1

# Frequency normalization
max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] /= max_freq

# Scoring Sentences
sentences = nltk.sent_tokenize(text)
sentences_score = {}

for sent in sentences:
    sent_words = preprocess_text(sent)
    for word in sent_words:
        if word in word_freq:
            if sent not in sentences_score:
                sentences_score[sent]  = word_freq[word]
            else:
                sentences_score[sent] += word_freq[word]

top_sentences = heapq.nlargest(3,
                               sentences_score,
                               key=sentences_score.get)

summary = " ".join(top_sentences)
print("Summarization:\n")
print(summary)

"""# Using TF-IDF"""

import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text

def summarize_text(text, num_sent=3):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess(sent) for sent in sentences]

    # Vectorization
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)

    # sentences scoring
    sentence_scores = {}
    for i, sent in enumerate(preprocessed_sentences):
        sentence_scores[sentences[i]] = np.mean(sentence_vectors[i].toarray())

    # Chose top sentences
    sorted_sentences = sorted(sentence_scores.items(),
                              key = lambda x: x[1],
                              reverse=True)
    summary = " ".join([sent[0] for sent in sorted_sentences[:num_sent]])
    return summary

summary = summarize_text(text)
print("Summarization:\n")
print(summary)

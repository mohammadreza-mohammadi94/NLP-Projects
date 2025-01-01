#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#


import os
import yaml
import pandas as pd
import numpy as np

"""# 2. Download Dataset"""

# git clone https://github.com/gunthercox/chatterbot-corpus.git

# Load dataset
PATH = r"/content/chatterbot-corpus/chatterbot_corpus/data/english"
files = [f for f in os.listdir(PATH) if f.endswith('.yml')]

"""## 2.1 Joining Data"""

conversations = []
labels = []

for file in files:
    with open(os.path.join(PATH, file), "r", encoding='utf-8') as f:
        data = yaml.safe_load(f)
        if 'conversations' in data:
            for conv in data['conversations']:
                if isinstance(conv, list) and len(conv) >= 2:
                    conversations.append(conv[0])
                    labels.append(conv[1])

"""# 3. Preprocessing"""

import os
import re
import nltk
import spacy

# python -m spacy download en_core_web_lg

nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_lg")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.is_alpha])

preprocessed_conversations = [preprocess(text) for text in conversations]
preprocessed_labels = [preprocess(text) for text in labels]

"""## 3.1 Vectorization"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer()
Q = vectorizer.fit_transform(preprocessed_conversations)
A = vectorizer.transform(labels)

from sklearn.metrics.pairwise import cosine_similarity

def get_response(user_input):
    user_input = preprocess(user_input)
    user_input_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_tfidf, A)
    best_match = similarities.argmax()
    return labels[best_match]

inp = 'Hello'
res = get_response(inp)
print(res)

inp = 'How are you'
res = get_response(inp)
print(res)

inp = 'what is AI'
res = get_response(inp)
print(res)


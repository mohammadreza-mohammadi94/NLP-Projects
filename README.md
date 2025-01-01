# 🗨️ Simple Chatter Bot  

This project demonstrates how to build a **simple chatbot** using **TF-IDF Vectorization** and **Cosine Similarity**. It uses conversational data from the **ChatterBot Corpus**.  

**Note:** This chatbot is a **basic example** and may produce incorrect or nonsensical responses. The main goal is to **illustrate the process of creating a chatbot**, not to build a production-ready model.  

---

## 📚 **Project Overview**  
1. **Dataset:** Uses conversational data from the [ChatterBot Corpus](https://github.com/gunthercox/chatterbot-corpus).  
2. **Preprocessing:** Text cleaning, tokenization, and lemmatization using **SpaCy** and **NLTK**.  
3. **Vectorization:** Converts text into numerical data using **TF-IDF Vectorizer**.  
4. **Response Generation:** Finds the best-matching response using **Cosine Similarity**.  

---

## 🛠️ **Requirements**  
- Python 3.x  
- NumPy  
- Pandas  
- SpaCy  
- NLTK  
- scikit-learn  
- PyYAML  

### Install Dependencies:  
```bash
pip install numpy pandas spacy nltk scikit-learn pyyaml
python -m spacy download en_core_web_lg
```

---

## 🚀 **How to Run the Chatbot**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/mohammadreza-mohammadi94/simple-chatterbot.git
   ```
2. Download the [ChatterBot Corpus](https://github.com/gunthercox/chatterbot-corpus):  
   ```bash
   git clone https://github.com/gunthercox/chatterbot-corpus.git
   ```
3. Update the dataset path in the code (`PATH`).  

---

## 🤖 **How it Works**  
1. Preprocesses user input and dataset using **SpaCy** and **NLTK**.  
2. Converts text into **TF-IDF vectors**.  
3. Measures **cosine similarity** to find the closest matching response.  
4. Returns the best match as a response.  

---

⚠️ **Disclaimer:** This chatbot is a simple demonstration and not intended for production use.  

This README is concise, clear, and aligns with your project's purpose. Let me know if you'd like to refine it further! 🚀

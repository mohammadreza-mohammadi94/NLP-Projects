# Fake News Classification Using LSTM

This project demonstrates the process of **fake news classification** using an **LSTM** (Long Short-Term Memory) model. The goal is to classify news articles as either **real** or **fake** based on their content. This is a crucial application in the field of Natural Language Processing (NLP), helping to automatically detect misinformation.

---

## Project Overview

- **Data Preprocessing**:  
  The project uses a dataset of labeled news articles and applies necessary preprocessing steps such as tokenization, padding, and text cleaning.

- **Model Architecture**:  
  The model is based on **LSTM** (Long Short-Term Memory) neural networks, which are designed to handle sequences of data, making them ideal for NLP tasks.

- **Model Training**:  
  The model is trained on the preprocessed dataset to classify news articles into fake or real categories.

- **Evaluation**:  
  The model is evaluated using various metrics like accuracy, precision, recall, and F1-score.

---

## Technologies Used

- **Python**: Programming language used for development.
- **TensorFlow/Keras**: Framework used for model building and training.
- **Natural Language Processing**: Techniques such as tokenization and padding used to process the text data.
- **LSTM**: A type of Recurrent Neural Network (RNN) used for sequence prediction tasks.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammadreza-mohammadi94/NLP-Projects
   ```

2. Navigate to the directory:
   ```bash
   cd NLP-Projects/Fake-News%20Classification%20Using%20LSTM
   ```

3. Open and run the notebook:
   ```bash
   Fake_News_Classification_LSTM_Model.ipynb
   ```

4. Ensure that the necessary libraries are installed:
   ```bash
   pip install numpy pandas tensorflow scikit-learn
   ```

5. Follow the steps in the notebook to preprocess the data, build and train the LSTM model, and evaluate its performance.

This project is aimed at providing a straightforward implementation of fake news classification using deep learning techniques, specifically LSTM. It's an excellent starting point for anyone interested in NLP and deep learning.

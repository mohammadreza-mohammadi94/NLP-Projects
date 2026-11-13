import streamlit as st

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

import nltk
nltk.download('punkt')
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from rouge_score import rouge_scorer


# Function to summarize raw text
def summarize_text(text, summarizer, num_sentences=3):
    """
    Summarizes the given text using the specified summarizer.

    Args:
        text (str): The raw text to be summarized.
        summarizer (sumy.summarizers.*): An instance of a summarizer from the Sumy library (e.g., LexRankSummarizer, TextRankSummarizer).
        num_sentences (int, optional): The number of sentences to include in the summary. Default is 3.

    Returns:
        str: A string containing the summarized text.
    """

    # Define Parser: Parses the input text using the PlaintextParser and Tokenizer from Sumy
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Generate summary: Uses the specified summarizer to create a summary of the text
    summary = summarizer(parser.document, num_sentences)
    # Convert summarized sentences to a single string
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text

# Function to evaluate ROUGE scores
def evaluate_rouge(reference_summary, generated_summary):
    """
    Evaluates the ROUGE scores between the reference summary and the generated summary.

    Args:
        reference_summary (str): The reference summary text to compare against.
        generated_summary (str): The generated summary text to evaluate.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    # Initialize ROUGE scorer with the desired metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def main():
    """
    Main function to run the Streamlit application for summarization.
    """

    st.title("Summarizer Application")

    # Creating Menu for slidebar
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Summarization Structure
    if choice == "Home":
        st.subheader("Summarization App")
        raw_text = st.text_area("Enter Text Here...")

        if st.button("Suumerize"):  # Submit button to summerize
            with st.expander("Original Text"):
                st.write(raw_text)
            
            # Designing Layout
            col1, col2 = st.columns(2)

            # LexRank Summarization
            with col1:
                with st.expander("LexRank Summary"):
                    lexrank_summarizer = LexRankSummarizer()
                    lexrank_summary = summarize_text(raw_text, lexrank_summarizer)
                    st.write(lexrank_summary)
                    if raw_text:
                        lex_score = evaluate_rouge(raw_text, lexrank_summary)
                        st.info("ROUGE Scores for LexRank Summary:")
                        st.dataframe(lex_score)

                        # Check Lenfth of Text
                        doc_len = {
                            "Original Text": len(raw_text),
                            "Summarized Text": len(lexrank_summary)
                        }
                        st.info("Length Of Text After/Before Summarization:")
                        st.dataframe(doc_len)


            # TextRank Summarization
            with col2:
                with st.expander("Text Rank Summary"):
                    textrank_summarizer = TextRankSummarizer()
                    textrank_summary = summarize_text(raw_text, textrank_summarizer)
                    st.write(textrank_summary) 
                    if raw_text:
                        text_score = evaluate_rouge(raw_text, textrank_summary)
                        st.info("ROUGE Scores for TextRank Summary:")
                        st.dataframe(text_score)

                        # Check Lenfth of Text
                        doc_len = {
                            "Original Text": len(raw_text),
                            "Summarized Text": len(textrank_summary)
                        }
                        st.info("Length Of Text After/Before Summarization:")
                        st.dataframe(doc_len)

    else:
        st.subheader("Application About")



if __name__ == '__main__':
    main()
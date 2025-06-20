
import streamlit as st
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sample_reviews = [
    "This product is amazing! I've never been happier.",
    "Terrible quality. Broke after one use.",
    "Pretty decent for the price. Would buy again.",
    "Not as described. Very disappointed.",
    "Fantastic! Exceeded expectations in every way.",
]

def get_sentiment(review):
    return TextBlob(review).sentiment.polarity

def summarize_reviews(reviews, num_sentences=3):
    full_text = " ".join(reviews)
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def generate_wordcloud(reviews):
    text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

st.title("📝 Product Review Summarizer")

st.subheader("Sentiment Analysis & Summarization")

st.write("## Sample Reviews:")
for review in sample_reviews:
    st.write(f"- {review}")

sentiments = [get_sentiment(r) for r in sample_reviews]
df = pd.DataFrame({"Review": sample_reviews, "Sentiment": sentiments})
st.write("### Sentiment Scores:")
st.dataframe(df)

st.write("### 🔍 Summary:")
st.success(summarize_reviews(sample_reviews))

st.write("### ☁️ Word Cloud:")
generate_wordcloud(sample_reviews)

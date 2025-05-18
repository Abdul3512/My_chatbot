import os
import json
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize

# SSL and NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk_data_path = os.path.abspath("nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Load intents from a JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocess data for ML model
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train TF-IDF vectorizer and Logistic Regression model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# Initialize Streamlit session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chatbot function using ML model
def chatbot_ml(input_text):
    vec = vectorizer.transform([input_text])
    tag = clf.predict(vec)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Chatbot function using simple pattern matching (fallback)
def chatbot_pattern(user_input):
    user_tokens = word_tokenize(user_input.lower())
    for intent in intents:
        for pattern in intent['patterns']:
            pattern_tokens = word_tokenize(pattern.lower())
            if set(pattern_tokens).intersection(user_tokens):
                return random.choice(intent['responses'])
    return chatbot_ml(user_input)  # fallback to ML if no pattern match

# Streamlit UI
def main():
    st.title("💬 Welcome To Your Number One Sport Chatbot")
    st.write("Ask me anything related to sports!")

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot_pattern(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")

if __name__ == "__main__":
    main()

import os
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Load intents from a JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocess data for ML model
tags = []
patterns = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
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
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

# Pattern-matching chatbot function (fallback)
def chatbot_pattern(user_input):
    user_tokens = re.findall(r"\w+", user_input.lower())  # NO NLTK
    for intent in intents["intents"]:
        for pattern in intent['patterns']:
            pattern_tokens = re.findall(r"\w+", pattern.lower())
            if set(pattern_tokens).intersection(user_tokens):
                return random.choice(intent['responses'])
    return chatbot_ml(user_input)

# Streamlit UI
def main():
    st.title("💬 Welcome To Your Number One Sport Chatbot")
    st.write("Ask me anything related to sports!")

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot_pattern(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")

if __name__ == "__main__":
    main()

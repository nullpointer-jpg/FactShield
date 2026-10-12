import streamlit as st
import joblib
import os
import re

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="FactShield - Fake News Detection",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model_path = os.path.join(
    MODEL_DIR,
    "factshield_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Header
# -----------------------------

st.title("🛡️ FactShield")

st.subheader("Intelligent Fake News Detection Using Machine Learning")

st.write(
    "FactShield analyzes the text of a news article and predicts "
    "whether it is likely to be REAL or FAKE using a machine-learning model."
)

st.divider()

# -----------------------------
# Input
# -----------------------------

st.subheader("Enter News Article")

news_text = st.text_area(
    "Paste the news article below:",
    height=250,
    placeholder="Paste the headline or complete news article here..."
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Check News", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    else:

        cleaned_text = clean_text(news_text)

        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        confidence = probability_dict[prediction] * 100

        st.divider()

        st.subheader("FactShield Analysis")

        if prediction == "REAL":

            st.success("✅ Prediction: REAL")

        else:

            st.error("⚠️ Prediction: FAKE")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.write(
            "The confidence score represents the model's estimated "
            "probability for its predicted class."
        )

        # Probability breakdown

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "REAL",
                f"{probability_dict.get('REAL', 0) * 100:.2f}%"
            )

        with col2:

            st.metric(
                "FAKE",
                f"{probability_dict.get('FAKE', 0) * 100:.2f}%"
            )

st.divider()

# -----------------------------
# About
# -----------------------------

st.subheader("About FactShield")

st.write(
    "FactShield is a machine-learning based fake news detection "
    "system developed as an academic project. The system uses "
    "TF-IDF for text feature extraction and Logistic Regression "
    "for classification."
)

st.caption(
    "Note: FactShield provides a machine-learning prediction and "
    "should not be treated as a definitive fact-checking service."
)
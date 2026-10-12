import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load datasets
fake_path = os.path.join(DATASET_DIR, "Fake.csv")
real_path = os.path.join(DATASET_DIR, "True.csv")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# Assign labels
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Combine title and article text
fake_df["content"] = (
    fake_df["title"].fillna("") + " " +
    fake_df["text"].fillna("")
)

real_df["content"] = (
    real_df["title"].fillna("") + " " +
    real_df["text"].fillna("")
)

# Combine datasets
df = pd.concat(
    [
        fake_df[["content", "label"]],
        real_df[["content", "label"]]
    ],
    ignore_index=True
)

# Clean missing and duplicate records
df["content"] = df["content"].astype(str).str.strip()

df = df[df["content"] != ""]
df = df.drop_duplicates(subset="content")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Total articles:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2),
    max_features=100000,
    sublinear_tf=True
)

print("\nConverting news articles into TF-IDF features...")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train machine-learning model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)

# Evaluate model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\n========== FACTSHIELD RESULTS ==========")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["FAKE", "REAL"]
))

# Save trained model and vectorizer
model_path = os.path.join(
    MODEL_DIR,
    "factshield_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

# Save evaluation metrics
metrics = {
    "total_articles": len(df),
    "training_articles": len(X_train),
    "testing_articles": len(X_test),
    "accuracy": round(accuracy, 4)
}

metrics_path = os.path.join(
    MODEL_DIR,
    "evaluation_metrics.json"
)

import json

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel training completed!")
print("Saved model:", model_path)
print("Saved vectorizer:", vectorizer_path)
print("Saved evaluation metrics:", metrics_path)
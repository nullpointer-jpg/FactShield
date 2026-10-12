# FactShield: Intelligent Fake News Detection Using Machine Learning

FactShield is a machine-learning based fake news detection system that analyzes news article text and predicts whether the article is likely to be REAL or FAKE.

## Features

- News article text classification
- TF-IDF based feature extraction
- Logistic Regression classification
- Prediction confidence score
- REAL and FAKE probability scores
- Interactive Streamlit web interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit

## Dataset

FactShield uses the Fake and Real News Dataset for training and evaluation.

Place the downloaded `Fake.csv` and `True.csv` files inside the `dataset/` directory before running `train_model.py`.

The dataset files are not included in this repository because of upload-size limitations.

After preprocessing and removing duplicate/empty articles:

- Total articles: 39,103
- Training articles: 31,282
- Testing articles: 7,821

## Model Performance

The Logistic Regression model achieved:

**Test Accuracy: 99.10%**

The evaluation was performed using a stratified 80/20 train-test split.

## How It Works

1. User enters a news article.
2. The text is cleaned.
3. TF-IDF converts the text into numerical features.
4. Logistic Regression analyzes the features.
5. FactShield predicts REAL or FAKE.
6. The application displays the prediction and probability scores.

## Running the Project

Install the required libraries:

```bash
pip install -r requirements.txt

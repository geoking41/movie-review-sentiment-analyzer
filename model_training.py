from gensim.models import FastText
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from utils import preprocess_text, review_to_vectors

# Load the IMDB dataset
data = pd.read_csv("imdb_dataset.csv")

# Preprocess reviews and convert sentiment to numerical values
data["review_preprocessed"] = data["review"].apply(preprocess_text)
data["sentiment_numeric"] = data["sentiment"].map({"negative": 0, "positive": 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["review_preprocessed"], data["sentiment_numeric"], test_size=0.2)

# Train FastText model
model = FastText(sentences=X_train, vector_size=100, window=5, min_count=1)

# Convert word vectors to a matrix
X_train_vectors = np.array([review_to_vectors(review, model) for review in X_train])
# X_test_vectors = [review_to_vectors(review) for review in X_test]

# Train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_vectors, y_train)

# Save the trained models
model.save("fasttext_model.bin")
joblib.dump(lr_model, "logistic_regression_model.pkl")

# Evaluate the model performance
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Predict sentiment on test data
# y_pred = model.predict(X_test_vectors)

# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 score:", f1)

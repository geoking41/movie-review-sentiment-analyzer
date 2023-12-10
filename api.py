from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim
from utils import preprocess_text, review_to_vectors
import joblib

app = Flask(__name__)
CORS(app)

# Load the models
model = gensim.models.FastText.load("fasttext_model.bin")
lr_model = joblib.load("logistic_regression_model.pkl")

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    # Get the review from the request body
    review = request.get_json()["review"]

    # Preprocess the review
    review_preprocessed = preprocess_text(review)

    # Convert review to vector representation
    review_vector = review_to_vectors(review_preprocessed, model)

    # Predict sentiment using the Logistic Regression model
    prediction = lr_model.predict([review_vector])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Calculate and include confidence score
    confidence_score = lr_model.predict_proba([review_vector])[:, prediction][0]

    # Prepare the response
    response = {
        "review": review,
        "sentiment": sentiment,
        "confidence_score": confidence_score
    }

    # Return the response as JSON
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
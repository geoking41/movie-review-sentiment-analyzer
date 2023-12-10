# movie-review-sentiment-analyzer

**For Beginner Learning:** AI model trained with Gensim and scikit-learn for classifying movie reviews as positive or negative

## Source Files

### API
API Service for review sentiment predicter

#### model_training.py
- helps train a FastText model with Logistic Regression using Gensim and scikit-learn with data from an IMDB data set found [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/)
- help generate the FastText model as a binary
- helps cache the Logistic Regression model
#### api.py
- helps consume the models generated above to serve the model predicter as an API (route: /predict) with flask
#### utils.py
- supporting file with important methods for tokenization and vectorization of text

### UI
Basic HTML UI with relevant styling and logic to invoke the API from local.
#### index.html
- provides a Text Area for user to provide a reivew
- on Submit, invokes the script.js
- shows the sentiment prediction and the confidence score as the result on successfull integration and execution of the API
#### style.css
- basic necessary styling for the UI
#### script.js
- helps fetch the review input from the UI and posts to the API
- receives the sentiment, confidence score from the API and returns response to the UI

## Setup

- clone the repo
- install packages using requirements.txt
- use nltk to download the stop words
- run the model_training.py. this is a time taking process (tried on Octa-Core AMD CPU with 16 GB RAM and a HDD)
- above generates the FastText and the Logistic Regression models probably after 10 - 15 mins. Hang On! :)
- now start the sentiment predictor service by running the api.py. serves on localhost:5000 by default
- now open up the index.html
- provide a review in the box on the ui and submit. Expectation is to view the sentiment and the confidence score

![image](https://github.com/geoking41/movie-review-sentiment-analyzer/assets/26020419/3de74355-a93b-4926-b4ae-926a767c7519)

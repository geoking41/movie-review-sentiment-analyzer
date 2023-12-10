import numpy as np
from gensim.utils import simple_preprocess

def review_to_vectors(review, model):
    vectors = []
    for token in review:
        try:
            vector = model.wv[token]
        except KeyError:
            pass
        else:
            vectors.append(vector)
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation and special characters
    # text = re.sub(r"[^a-zA-Z0-9\s]+", "", text)
    # Tokenize the text
    tokens = simple_preprocess(text)
    # Remove stop words (optional)
    # tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

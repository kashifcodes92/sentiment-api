import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example dummy data
data = pd.DataFrame({
    'text': ['I love this!', 'This is terrible', 'Amazing work', 'Worst experience'],
    'label': ['positive', 'negative', 'positive', 'negative']
})

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
print("Model trained and saved as sentiment_model.pkl")
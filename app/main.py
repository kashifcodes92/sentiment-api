from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Define input format
class TextInput(BaseModel):
    text: str

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is live!"}

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(input: TextInput):
    text_vector = vectorizer.transform([input.text])
    prediction = model.predict(text_vector)[0]
    return {"sentiment": prediction}

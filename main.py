# SIMPLE AI - SENTIMENT ANALYSIS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data (examples)
texts = [
    "I love this movie, it was amazing!",
    "This product is terrible, I hate it",
    "The food was delicious, I will come again",
    "Worst experience ever",
    "The game is fun and exciting",
    "This is boring and bad"
]

labels = [1, 0, 1, 0, 1, 0]   # 1 = positive, 0 = negative

# Convert text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the AI model
model = LogisticRegression()
model.fit(X, labels)

# Function: Predict sentiment
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Positive 🙂" if prediction == 1 else "Negative 😟"

# Ask the user for input
while True:
    user_text = input("Type a sentence (or 'quit'): ")
    if user_text == "quit":
        break
    print("Sentiment:", predict_sentiment(user_text))

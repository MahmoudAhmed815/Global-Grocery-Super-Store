import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem import SnowballStemmer

# Load data
df = pd.read_csv('cleaned_data.csv', encoding="ISO-8859-1")

# Combine useful columns
rec = df[['Category', 'Sub-Category', 'Product Name']].copy()
rec['text'] = rec['Category'] + " " + rec['Sub-Category'] + " " + rec['Product Name']
rec = rec.drop_duplicates(subset=['text']).reset_index(drop=True)

# Text cleaning
stemmer = SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]+', ' ', text).strip()
    return " ".join(stemmer.stem(word) for word in text.split())

rec['clean_text'] = rec['text'].apply(clean)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(rec['clean_text'])

# Model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(X)

# Save everything
with open('product_rec.pkl', 'wb') as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "model": model,
        "data": rec,
    }, f)

print("✅ Model trained and saved as product_rec.pkl")
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('diseases_with_symptoms.csv')
df.dropna(subset=['Symptoms'], inplace=True)  

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Symptoms'])

from sklearn.metrics.pairwise import cosine_similarity

def predict_diseases(user_input, top_n=5):
    user_vec = vectorizer.transform([user_input])
    cos_sim = cosine_similarity(user_vec, X).flatten()
    top_indices = cos_sim.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        disease = df.iloc[idx]['Disease Name']
        prob = cos_sim[idx]
        results.append((disease, f"{prob:.2f}"))
    return results

user_symptoms = "high fever, sore throat"
predictions = predict_diseases(user_symptoms)
print(predictions)  
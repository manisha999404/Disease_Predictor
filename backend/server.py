from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from gemini_helper import get_disease_details

app = Flask(__name__)

# --- Load dataset ---
df = pd.read_csv('Databases/final_diseases_with_symptoms_enhanced.csv')
df['Symptoms'] = df['Symptoms'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

# --- TF-IDF Model ---
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(df['Symptoms'].apply(lambda x: ' '.join(x)))

def predict_diseases(user_symptoms, top_n=5):
    user_vector = vectorizer.transform([' '.join(user_symptoms)])
    similarities = cosine_similarity(user_vector, symptom_vectors).flatten()
    adjusted_scores = similarities * (1 - 0.2 * df['IsRare'])  # rare disease penalty

    ranked_indices = adjusted_scores.argsort()[::-1][:top_n]
    results = []

    for idx in ranked_indices:
        disease = df.iloc[idx]['Disease Name']
        prob = adjusted_scores[idx] / adjusted_scores[ranked_indices].sum()

        # --- Get details from Gemini ---
        details = get_disease_details(disease)

        results.append({
            "disease": disease,
            "probability": f"{prob:.1%}",
            **details
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('symptoms', '')
    user_symptoms = [s.strip().lower() for s in user_input.split(',') if s.strip()]

    predictions = predict_diseases(user_symptoms, top_n=3)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

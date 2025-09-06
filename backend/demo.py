from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import os

app = Flask(__name__)
df = pd.read_csv('Databases/final_diseases_with_symptoms_enhanced.csv')
df['Symptoms'] = df['Symptoms'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(df['Symptoms'].apply(lambda x: ' '.join(x)))

session_data = {
    'user_symptoms': [],
    'top_diseases': [],
    'follow_up_symptoms': [],
    'current_question_index': 0
}

def predict_diseases(user_symptoms, top_n=5):
    user_vector = vectorizer.transform([' '.join(user_symptoms)])
    similarities = cosine_similarity(user_vector, symptom_vectors).flatten()
    adjusted_scores = similarities * (1 - 0.2 * df['IsRare'])
    ranked_indices = adjusted_scores.argsort()[::-1][:top_n]
    ranked_diseases = df.iloc[ranked_indices]['Disease Name'].tolist()
    ranked_scores = adjusted_scores[ranked_indices]
    probabilities = ranked_scores / ranked_scores.sum()
    return list(zip(ranked_diseases, probabilities))

def ask_follow_up_questions(user_symptoms, top_diseases, max_questions=10):
    follow_up_symptoms = set()
    for disease in top_diseases:
        symptoms = df[df['Disease Name'] == disease]['Symptoms'].iloc[0]
        follow_up_symptoms.update(symptoms)
    follow_up_symptoms -= set(user_symptoms)
    follow_up_symptoms = list(follow_up_symptoms)[:max_questions]
    return follow_up_symptoms

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    global session_data
    session_data = {
        'user_symptoms': [],
        'top_diseases': [],
        'follow_up_symptoms': [],
        'current_question_index': 0
    }
    
    data = request.json
    user_input = data.get('symptoms', '')
    
    user_symptoms = [s.strip().lower() for s in user_input.split(',')]
    session_data['user_symptoms'] = user_symptoms
    
 
    prediction_results = predict_diseases(user_symptoms, top_n=5)
    session_data['top_diseases'] = [disease for disease, _ in prediction_results]
    

    session_data['follow_up_symptoms'] = ask_follow_up_questions(
        user_symptoms, 
        session_data['top_diseases'], 
        max_questions=10
    )
    
    predictions = []
    for disease, probability in prediction_results:
        predictions.append({
            'disease': disease,
            'probability': f"{probability:.1%}"
        })
    
    return jsonify(predictions)

@app.route('/next_question', methods=['POST'])
def next_question():
    global session_data
    data = request.json
    answer = data.get('answer')
    
    if answer is not None:
        current_symptom = session_data['follow_up_symptoms'][session_data['current_question_index'] - 1]
        if answer.lower() == 'yes':
            session_data['user_symptoms'].append(current_symptom)
    

    if session_data['current_question_index'] < len(session_data['follow_up_symptoms']):
        next_symptom = session_data['follow_up_symptoms'][session_data['current_question_index']]
        session_data['current_question_index'] += 1
        return jsonify({"question": f"Do you have {next_symptom}?"})
    else:
   
        final_predictions = predict_diseases(session_data['user_symptoms'], top_n=1)
        final_disease, probability = final_predictions[0]
        return jsonify({"final_disease": f"{final_disease} ({probability:.1%})"})

if __name__ == '__main__':
    app.run(debug=True)
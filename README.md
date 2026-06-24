# AURA — AI-Driven Disease Prediction & Solution Platform

> Predict diseases from symptoms, powered by ML and Gemini AI.

**Live Demo:** [https://disease-predictor-tan.vercel.app/](https://disease-predictor-tan.vercel.app/) &nbsp;·&nbsp; **Repo:** [github.com/manisha999404/Disease_Predictor](https://github.com/manisha999404/Disease_Predictor)

---

## Overview

AURA is a full-stack health companion that takes a set of symptoms as input and returns the top predicted diseases ranked by probability, along with AI-generated prevention tips, home remedies, and specialist recommendations for each result. The interface is anchored by an animated canvas assistant — AURA — with real-time cursor-tracking eyes and ambient interactions.

---

## Features

- **Symptom-based disease prediction** — TF-IDF vectorization + cosine similarity over a custom-scraped disease-symptom dataset returns the top 3 ranked predictions with probability scores
- **Gemini AI integration** — each prediction is enriched with contextual prevention advice, home remedies, risk level, and specialist type via the Gemini API
- **Animated AI assistant** — canvas-rendered AURA character with cursor-tracking eyes, blob animation, and a 30fps throttled render loop for performance
- **Responsive UI** — built with React, TypeScript, and Tailwind CSS; works across desktop and mobile
- **REST API backend** — Flask server with CORS support, lazy model loading, and gunicorn WSGI for production

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React 19, TypeScript, Vite, TanStack Start, Tailwind CSS |
| Backend | Python, Flask, flask-cors, gunicorn |
| ML / Data | Scikit-learn, TF-IDF, Pandas, BeautifulSoup |
| AI | Google Gemini API |
| Deployment | Vercel (frontend), Render (backend) |

---

## Project Structure

```
Disease_Predictor/
├── frontend/               # React + Vite + TanStack Start
│   ├── src/
│   │   ├── components/     # AuraAssistant, BackgroundEffects, UI components
│   │   ├── routes/         # index.tsx — main symptom input + results page
│   │   └── lib/            # API helpers, utilities
│   └── .env                # FLASK_API_URL
│
└── backend/                # Flask REST API
    ├── server.py            # /predict endpoint
    ├── gemini_helper.py     # Gemini API integration
    ├── Databases/           # Disease-symptom CSV datasets
    └── .env                 # GEMINI_API_KEY
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+

### Backend

```bash
cd backend
pip install -r requirements.txt

# Create .env
echo "GEMINI_API_KEY=your_key_here" > .env

python server.py
# Runs on http://127.0.0.1:5000
```

### Frontend

```bash
cd frontend
npm install

# Create .env
echo "FLASK_API_URL=http://127.0.0.1:5000" > .env

npm run dev
# Runs on http://localhost:3000
```

---

## How It Works

1. User enters symptoms (e.g. "fever, cough, fatigue") into the AURA interface
2. Frontend sends symptoms to the Flask `/predict` endpoint
3. Backend vectorizes the input using TF-IDF and computes cosine similarity against the disease-symptom dataset
4. Top 3 matching diseases are returned with probability scores
5. Gemini API enriches each result with prevention tips, remedies, specialist type, and risk level
6. Results are rendered in the UI alongside the AURA assistant

---

## API Reference

### `POST /predict`

**Request**
```json
{
  "symptoms": "fever, cough, fatigue"
}
```

**Response**
```json
[
  {
    "disease": "Influenza",
    "probability": "78.3%",
    "risk": "Moderate",
    "prevention": ["Stay hydrated", "Rest adequately"],
    "remedies": ["Ginger tea", "Steam inhalation"],
    "specialist": "General Physician"
  }
]
```

---

## Deployment

| Service | Purpose | Config |
|---|---|---|
| Vercel | Frontend hosting | `FLASK_API_URL` env var |
| Render | Backend hosting | `GEMINI_API_KEY` env var, gunicorn start command |

---

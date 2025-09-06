import google.generativeai as genai
import os
import json

# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Cache to avoid duplicate API calls ---
CACHE = {}

def get_disease_details(disease_name: str):
    if disease_name in CACHE:
        return CACHE[disease_name]

    prompt = f"""
    You are a medical assistant. 
    Task: Return structured health advice for the disease: {disease_name}.

    Requirements:
    - Output ONLY valid JSON (no explanations, no markdown).
    - Keys must be exactly: prevention, remedies, specialist, risk

    Instructions for each key:
    - "prevention": 3-5 short prevention tips (diet, lifestyle, hygiene).
    - "remedies": 2-4 safe home remedies for mild symptoms.
    - "specialist": the type of doctor or specialist to consult.
    - "risk": one word among ["Mild", "Moderate", "Severe"].

    Example format:
    {{
      "prevention": ["Tip1", "Tip2"],
      "remedies": ["Remedy1", "Remedy2"],
      "specialist": "Specialist Name",
      "risk": "Mild"
    }}
    """

    try:
        response = model.generate_content(prompt)

        # --- Debug raw response ---
        print("RAW GEMINI RESPONSE:", response)

        # --- Safely extract text ---
        text = None
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                text = parts[0].text

        if not text:
            raise ValueError("Empty Gemini response")

        # --- Clean accidental formatting ---
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()

        # --- Parse JSON ---
        details = json.loads(text)

    except Exception as e:
        print("Gemini error:", e)  # log actual error
        details = {
            "prevention": ["Data not available"],
            "remedies": ["Data not available"],
            "specialist": "General Physician",
            "risk": "Moderate"
        }

    CACHE[disease_name] = details
    return details

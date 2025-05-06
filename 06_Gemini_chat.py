import joblib
import shap
import numpy as np
import google.generativeai as genai

genai.configure(api_key="AIzaSyAoj6Z8jzJ71wb1Qis5F7hU-nvKHjBDlv0")

model = joblib.load('linear_svm_model.pkl')
scaler = joblib.load('scaler.pkl')
X_train_sample = joblib.load('X_train_sample.pkl')
feature_names = ['Age', 'Sex', 'BMI', 'HighBP', 'PhysActivity', 'Smoker', 'GenHlth']

chat_model = genai.GenerativeModel("gemini-1.5-flash")
chat = chat_model.start_chat(history=[])

def explain_prediction(model, patient_scaled, feature_names, original_input):
    explainer = shap.LinearExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(patient_scaled)
    explanation = []
    for i, val in enumerate(shap_values[0]):
        direction = "increased" if val > 0 else "decreased"
        explanation.append(f"Feature '{feature_names[i]}' with value {original_input[i]} {direction} the diabetes risk by {abs(val):.3f}.")
    return explanation

def get_initial_gemini_response(prediction, explanation_list):
    prompt = f"""
You are a health assistant AI. A model predicted that a patient is {'diabetic' if prediction == 1 else 'not diabetic'}.
Here are the explanation points from SHAP values:

{chr(10).join(['- ' + e for e in explanation_list])}

Please explain the diagnosis in simple terms like a doctor talking to a patient.
"""
    response = chat.send_message(prompt)
    return response.text


def get_followup_response(question):
    response = chat.send_message(question)
    return response.text

def diabetes_chatbot():
    print("🤖 Hello! I'm DiaBot, your diabetes prediction assistant.")
    print("Please enter patient details:\n")

    prompts = [
        "Age (in 5-year bins e.g., 1=18-24, ..., 13=80+): ",
        "Sex (0=Female, 1=Male): ",
        "BMI (e.g., 25.3): ",
        "HighBP (0=No, 1=Yes): ",
        "PhysActivity (0=No, 1=Yes): ",
        "Smoker (0=No, 1=Yes): ",
        "General Health (1=Excellent to 5=Poor): "
    ]

    patient_input = [float(input(p)) for p in prompts]
    patient_scaled = scaler.transform([patient_input])
    prediction = model.predict(patient_scaled)[0]
    explanation = explain_prediction(model, patient_scaled, feature_names, patient_input)

    print("\n🤖 Fetching explanation from Gemini...\n")
    gemini_response = get_initial_gemini_response(prediction, explanation)
    print("💬 Gemini says:\n")
    print(gemini_response)

    while True:
        follow_up = input("\n💬 Ask a follow-up question (or type 'exit' to end): ")
        if follow_up.lower() == 'exit':
            print("👋 Goodbye!")
            break
        try:
            response = get_followup_response(follow_up)
            print("🤖", response)
        except Exception as e:
            print("⚠️ Error during Gemini response:", e)

if __name__ == "__main__":
    diabetes_chatbot()
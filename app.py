import streamlit as st
import joblib
import shap
import numpy as np
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyAoj6Z8jzJ71wb1Qis5F7hU-nvKHjBDlv0")  # Replace with your Gemini API key

# Load ML model and scaler
model = joblib.load("linear_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
X_train_sample = joblib.load("X_train_sample.pkl")
feature_names = ['Age', 'Sex', 'BMI', 'HighBP', 'PhysActivity', 'Smoker', 'GenHlth']

# Setup Gemini chat
if "chat" not in st.session_state:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    st.session_state.chat = gemini_model.start_chat(history=[])

# Setup memory for prediction results
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.explanation = None
    st.session_state.gemini_response = ""

def get_age_range(age_bin):
    age_ranges = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
        11: "70-74", 12: "75-79", 13: "80+"
    }
    return age_ranges.get(age_bin, "Unknown")


# SHAP explanation
def explain_prediction(model, patient_scaled, original_input):
    explainer = shap.LinearExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(patient_scaled)
    explanation = []
    for i, val in enumerate(shap_values[0]):
        direction = "increased" if val > 0 else "decreased"
        explanation.append(
            f"Feature '{feature_names[i]}' with value {original_input[i]} {direction} the diabetes risk by {abs(val):.3f}."
        )
    return explanation

def get_initial_gemini_response(prediction, explanation_list, patient_input):
    readable_age = get_age_range(int(patient_input[0]))
    prompt = f"""
You are a helpful medical assistant AI. A model predicted that a patient is {'diabetic' if prediction == 1 else 'not diabetic'}.
Here are some factors and explanations for the decision:

{chr(10).join(['- ' + e for e in explanation_list])}

The age was provided as bin index {int(patient_input[0])}, which corresponds to age group: {readable_age}.

Now, explain this result in simple and friendly terms to a patient, as a doctor would. Be empathetic, explain gently, and clearly describe which factors increased or decreased risk.
Do NOT include any greetings or patient names in your response.
"""
    return st.session_state.chat.send_message(prompt).text



# App UI
st.set_page_config(page_title="Diabetes AI Assistant", layout="centered")
st.title("Diabetes Risk Predictor")

st.subheader("Enter patient details:")

with st.form("patient_form"):
    age = st.slider("Age group (1 = 18-24, ..., 13 = 80+)", 1, 13, 5)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    bmi = st.number_input("BMI", 10.0, 60.0, step=0.1)
    highbp = st.radio("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    phys = st.radio("Physical Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    smoker = st.radio("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    genhlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)

    submitted = st.form_submit_button("🔍 Predict")

# On form submission
if submitted:
    patient_input = [age, sex, bmi, highbp, phys, smoker, genhlth]
    scaled_input = scaler.transform([patient_input])
    prediction = model.predict(scaled_input)[0]
    explanation = explain_prediction(model, scaled_input, patient_input)

    # Store results in session state
    st.session_state.prediction = prediction
    st.session_state.explanation = explanation

    with st.spinner("🤖 Asking Gemini for a friendly explanation..."):
        gemini_response = get_initial_gemini_response(prediction, explanation, patient_input)
        st.session_state.gemini_response = gemini_response

# Display stored prediction and explanation
if st.session_state.prediction is not None:
    st.success(f"✅ Prediction: The patient is **{'Diabetic' if st.session_state.prediction == 1 else 'Not Diabetic'}**.")
    st.write("### Explanation")
    st.markdown(st.session_state.gemini_response)

    # Follow-up Q&A
    st.write("### Follow-up Questions")
    followup_question = st.text_input("Your question:")
    if followup_question:
        with st.spinner("🧠 Thinking..."):
            followup_response = st.session_state.chat.send_message(followup_question)
            st.markdown(f"**Gemini:** {followup_response.text}")

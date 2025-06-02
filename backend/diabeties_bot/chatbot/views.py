from django.shortcuts import render
from .forms import PatientFeatureForm
from .shap_explainer import explain_prediction
from .langchain_bot import talk_to_bot
from .logic import predict

FEATURE_NAMES = [
    "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "Smoker", 
    "PhysActivity", "DiffWalk", "BMI", "MentHlth", "PhysHlth", "GenHlth", 
    "Sex", "Age", "HvyAlcoholConsump"
]

def chatbot_view(request):
    form = PatientFeatureForm()
    explanation = []
    response = ""
    prediction = None
    bot_followup = None

    if request.method == "POST":
        if "user_message" in request.POST:
            # 🎯 Handle follow-up question
            user_msg = request.POST.get("user_message")
            prediction = request.session.get("prediction", "No prediction found.")
            explanation = request.session.get("explanation", [])
            followup_input = f"Prediction: {prediction}\nExplanation:\n" + "\n".join(explanation) + f"\nFollow-up: {user_msg}"
            bot_followup = talk_to_bot(followup_input, prediction)
        else:
            # 🩺 Handle main form submission
            form = PatientFeatureForm(request.POST)
            if form.is_valid():
                try:
                    features = [
                        int(form.cleaned_data['HighBp']),
                        int(form.cleaned_data['Highchol']),
                        int(form.cleaned_data['HeartDiseaseorAttack']),
                        int(form.cleaned_data['Stroke']),
                        int(form.cleaned_data['Smoker']),
                        int(form.cleaned_data['PhysActivity']),
                        int(form.cleaned_data['DiffWalk']),
                        float(form.cleaned_data['Bmi']),
                        int(form.cleaned_data['MentHlth']),
                        int(form.cleaned_data['PhysHlth']),
                        int(form.cleaned_data['GenHlth']),
                        int(form.cleaned_data['Sex']),
                        int(form.cleaned_data['Age']),
                        int(form.cleaned_data['HvyAlcoholConsump']),
                    ]

                    prediction_label = predict(features)
                    shap_values = explain_prediction(features)

                    if prediction_label[0] is not None and shap_values is not None:
                        prediction = "The patient is Diabetic or pre-diabetic" if prediction_label == 1 else "Patient is Not Diabetic"
                        explanation = shap_values
                        response = talk_to_bot("The explanation is:\n" + "\n".join(explanation), prediction)

                        # Save for follow-up
                        request.session['prediction'] = prediction
                        request.session['explanation'] = explanation
                    else:
                        prediction = "Prediction failed. Please check your input."
                        response = "Something went wrong during prediction. Check input format."

                except Exception as e:
                    prediction = "Error occurred during processing."
                    response = f"An error occurred: {str(e)}"
                    explanation = []

    else:
        # Populate existing data if available
        prediction = request.session.get("prediction", None)
        explanation = request.session.get("explanation", [])

    return render(request, "chat.html", {
        "form": form,
        "prediction": prediction,
        "explanation": explanation,
        "response": response,
        "bot_followup": bot_followup
    })

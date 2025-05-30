from django.shortcuts import render
from .forms import PatientFeatureForm
from .shap_explainer import explain_prediction
from .langchain_bot import talk_to_bot
from .logic import predict

FEATURE_NAMES = [
    "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "Smoker", 
    "PhysActivity", "DiffWalk", "BMI", "MentHlth", "PhysHlth", "GenHlth", 
    "Sex", "Age", "HvyAlcoholConsump"]


def chatbot_view(request):
    form = PatientFeatureForm()
    explanation = []
    response = ""
    prediction = None

    if request.method == "POST":
        form = PatientFeatureForm(request.POST)
        if form.is_valid():
            try:
                # Safely parse and cast inputs to correct types
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

                # Ensure valid prediction
                prediction_label = predict(features)
                print(prediction_label[0])
                shap_values=explain_prediction(features)
                print(shap_values)
                if prediction_label[0] is not None and shap_values is not None:
                    prediction = "Diabetic" if prediction_label == 1 else "Not Diabetic"
                    explaination=shap_values

                    response = talk_to_bot(
                        f"The model predicted: {prediction}. The explanation is:\n" +
                        "\n".join(explanation)
                    )
                else:
                    prediction = "Prediction failed. Please check your input."
                    response = "Something went wrong during prediction. Check input format."

            except Exception as e:
                prediction = "Error occurred during processing."
                response = f"An error occurred: {str(e)}"
                explanation = []

    return render(request, "chat.html", {
        "form": form,
        "prediction": prediction,
        "explanation": explanation,
        "response": response
    })

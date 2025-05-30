import shap
import numpy as np
from joblib import load
import pandas as pd
from sklearn.impute import SimpleImputer

# Load model, scaler, and data
model = load(r"C:\Users\slowd\OneDrive\Desktop\Projects\MLProjects\Diabeties\final_model.pkl")
scaler = load(r"C:\Users\slowd\OneDrive\Desktop\Projects\MLProjects\Diabeties\scaler.pkl")
x_shap = pd.read_csv(r'C:\Users\slowd\OneDrive\Desktop\Projects\MLProjects\Diabeties\x_shap.csv')

# Define correct feature order
FEATURE_ORDER = [
    "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke",
    "Smoker", "PhysActivity", "DiffWalk", "BMI", "MentHlth",
    "PhysHlth", "GenHlth", "Sex", "Age", "HvyAlcoholConsump"
]

# Prepare SHAP

# Prediction function
def predict(input_dict):
    try:
        # Ensure correct order and types
        print("User Input:", input_dict)
        print(type(input_dict))
        print("check 1")

        print("Data types:", [type(x) for x in input_dict])

        # Impute and scale
        print("check 2")
        user_input_scaled = scaler.transform([input_dict])
        print("Transformed Input:", [user_input_scaled])
        print("check 3")

        # Predict and explain
        prediction = model.predict(user_input_scaled)
        print("Prediction:", prediction)

        return prediction
    except Exception as e:
        print("Prediction failed:", str(e))
        return None, None

import os
import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Absolute path to ML files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print
MODEL_DIR = os.path.join(BASE_DIR, '..', '..')  # One level up from `backend`

model = joblib.load(os.path.join(MODEL_DIR, "final_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

x_shap = pd.read_csv(os.path.join(MODEL_DIR, "x_shap.csv"))
print("check 1")
  # 1st sample, class 1 SHAP values

imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_shap)
x_train_scaled = scaler.transform(x_train_imputed)
background = shap.kmeans(x_train_scaled, 250)
explainer = shap.KernelExplainer(model.predict_proba, background)
print("check 2")

FEATURE_NAMES = ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "Smoker",
                 "PhysActivity", "DiffWalk", "BMI", "MentHlth", "PhysHlth",
                 "GenHlth", "Sex", "Age", "HvyAlcoholConsump"]

def explain_prediction(patient_features):
  
    x = np.array(patient_features).reshape(1, -1)
    x_scaled = scaler.transform(imputer.transform(x))
    print("check 3")
    # Get SHAP values
    shap_values = explainer.shap_values(x_scaled)
    print("check 4")
    print("shap val:", shap_values)
    actual_list=shap_values[0]
    print("actual values:", actual_list)
    class1_values=[]
    for i in actual_list:
      class1_values.append(i[1])
    
    print(class1_values)
    result = []
    for i in range(len(class1_values)):
        direction = "increased" if  class1_values[i]> 0 else "decreased" if class1_values[i] < 0 else "had no effect on"
        result.append(f"**{FEATURE_NAMES[i]}** {direction} the likelihood of being diabetic")
    print("check 5")
    print(result)
    return result




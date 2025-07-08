import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print
MODEL_DIR = os.path.join(BASE_DIR, '..', '..')

model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
FEATURE_NAMES = ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "Smoker",
                 "PhysActivity", "DiffWalk", "BMI", "MentHlth", "PhysHlth",
                 "GenHlth", "Sex", "Age", "HvyAlcoholConsump"]

def explain_prediction(patient_features):
    if len(patient_features) != len(FEATURE_NAMES):
      raise ValueError("Input feature count does not match model expectation.")
    x = np.array(patient_features).reshape(1, -1)
    print("check 3")
    
    shap_values = explainer.shap_values(x)
    print("check 4")
    print("shap val:", shap_values)
    
    class1_values = shap_values[0]
    print("actual values:", class1_values)

    result = []
    for i in range(len(class1_values)):
        sv = class1_values[i]
        val = patient_features[i]
        feature = FEATURE_NAMES[i]

        if sv <= 0:
            continue
        
        # Filter logic
        if feature in ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "Smoker", "PhysActivity", "DiffWalk", "HvyAlcoholConsump"]:
            if val == 1:
                result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "BMI" and val > 30:
            result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "MentHlth" and val > 5:
            result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "PhysHlth" and val > 5:
            result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "GenHlth" and val >= 4:
            result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "Age" and val >= 9:
            result.append(f"{feature} increased the likelihood of being diabetic")
        elif feature == "Sex" and val == 1:
            result.append(f"{feature} increased the likelihood of being diabetic")

    print("check 5")
    print(result)

    return result

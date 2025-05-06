import joblib
import numpy as np

model = joblib.load('linear_svm_model.pkl')
scaler = joblib.load('scaler.pkl')

def chatbot():
    print("Welcome to the Diabetes Prediction Chatbot!")
    print("Please answer the following questions to check if you're at risk of diabetes.")
    
    try:
        age = float(input("Enter your age: "))
        sex = int(input("Sex (1 for Male, 0 for Female): "))
        bmi = float(input("Enter your BMI: "))
        highbp = int(input("Do you have High Blood Pressure? (1 for Yes, 0 for No): "))
        physactivity = int(input("Do you do physical activity? (1 for Yes, 0 for No): "))
        smoker = int(input("Are you a smoker? (1 for Yes, 0 for No): "))
        genhlth = int(input("How would you rate your general health? (1 to 5 scale): "))
    except ValueError:
        print("Invalid input! Please enter numeric values where applicable.")
        return

    patient_input = np.array([age, sex, bmi, highbp, physactivity, smoker, genhlth]).reshape(1, -1)

    patient_input_scaled = scaler.transform(patient_input)

    prediction = model.predict(patient_input_scaled)

    if prediction[0] == 1:
        print("⚠️ You are likely diabetic.")
    else:
        print("✅ You are likely not diabetic.")

if __name__ == "__main__":
    chatbot()

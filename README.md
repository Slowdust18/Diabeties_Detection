# Diabetes Risk Prediction and Explanation App

This is a Django-based web application that allows users to input basic health-related features to **predict the risk of diabetes** using an **XGBoost model** and **explain the prediction** using **SHAP values** and **natural language generation via LangChain**.

---

## ✨ Features

- 🧠 **Predicts diabetes risk** using a trained XGBoost model.
- 📊 **Explains the prediction** with SHAP values (feature-level contribution).
- 🗣️ **Conversational interface** for follow-up questions using LangChain + Gemini Pro.
- ✅ Easy-to-use form interface for patients or general users.
- 📦 Model and explainer are pre-loaded for fast inference.

---

## 📂 Project Structure
These are the only files that are used the remainder were for 
```
backend/
│
├── chatbot/
│   ├── views.py               # Main view logic
│   ├── forms.py               # Django form for user input
│   ├── logic.py               # Prediction logic using XGBoost
│   ├── shap_explainer.py      # SHAP explanation logic
│   ├── langchain_bot.py       # LLM-based natural language explanation
│
├── templates/
│   └── chat.html              # Frontend template
    └── form.html
│
├── xgb_model.pkl            # Trained XGBoost model
├── shap_explainer.pkl       # Precomputed SHAP explainer
├── Final_Training.ipynb     # The training procedure
```

---

## 📋 Input Features

| Feature Name        | Type     | Description |
|---------------------|----------|-------------|
| `HighBP`            | Binary   | 0 = No, 1 = Yes |
| `HighChol`          | Binary   | 0 = No, 1 = Yes |
| `HeartDiseaseorAttack` | Binary | 0 = No, 1 = Yes |
| `Stroke`            | Binary   | 0 = No, 1 = Yes |
| `Smoker`            | Binary   | 0 = No, 1 = Yes |
| `PhysActivity`      | Binary   | 0 = No, 1 = Yes |
| `DiffWalk`          | Binary   | 0 = No, 1 = Yes |
| `BMI`               | Float    | Body Mass Index |
| `MentHlth`          | Integer  | Days of poor mental health (last 30 days) |
| `PhysHlth`          | Integer  | Days of poor physical health (last 30 days) |
| `GenHlth`           | Integer  | 1 = Excellent to 5 = Poor |
| `Sex`               | Binary   | 0 = Female, 1 = Male |
| `Age`               | Categorical | Age group (1 = 18–24 to 13 = 80+) |
| `HvyAlcoholConsump` | Binary   | 0 = No, 1 = Yes |

---

## 🧪 Setup Instructions

### 1. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

**Required:**
- `xgboost`
- `shap`
- `scikit-learn`
- `joblib`
- `django`
- `langchain`
- `langchain-google-genai`
- `python-dotenv`

### 2. 🧠 Environment setup

Create a `.env` file:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. 🚀 Run the app

```bash
python manage.py runserver
```

Visit: `http://127.0.0.1:8000/`

---

## 🧠 How It Works

1. 📝 User submits form with personal health features.
2. 🤖 Model (`XGBoost`) makes a binary prediction (diabetic or not).
3. 📈 `SHAP` values identify which features contributed to the prediction.
4. 💬 LangChain generates natural language explanations using Gemini Pro.
5. 🗨️ Users can ask follow-up questions and get contextual responses.

---

## 🔮 Future Improvements

- SHAP force/waterfall plot rendering.
- Doctor recommendation cards.
- Upload .csv batch prediction.
- Multilingual support.

---

## 🛡️ License

MIT License

---

## 👨‍💻 Author

**Anirudh Sarkar**  
AI/ML Enthusiast | SIH 2024 Winner  
Always open to collaboration!

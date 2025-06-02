
from langchain_google_genai import GoogleGenerativeAI

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

print("GEMINI_API_KEY loaded:", os.getenv("GEMINI_API_KEY") is not None)

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))


template = """
You are a helpful medical assistant. Based on the prediction explanation below,
explain in simple language what the SHAP outputs mean to the patient. From a list of these features 
HighBP, 
HighChol, 
HeartDiseaseorAttack, 
Stroke, 
Smoker,
PhysActivity, 
DiffWalk, 
BMI, 
MentHlth, 
PhysHlth,
GenHlth
Sex
Age
HvyAlcoholConsump
The prediction is : {prediction}
Only the features contributing toward diabeties are sent not all features are sent if the features are not sent it means they did not contribute towards the patient being diabetic
SHAP Explanation:
{explanation}

Do not refer to patient as with any names just give precise and consise explaination for the issue and the follow up questions.
Give general health advice nothing too specific do not keep repeating the same things.
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

def talk_to_bot(explanation_list, prediction):
    explanation = "\n".join(explanation_list)
    return chain.run(explanation=explanation, prediction=prediction)
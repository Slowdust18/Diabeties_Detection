
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
explain in simple language what the SHAP outputs mean to the patient.

SHAP Explanation:
{explanation}
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

def talk_to_bot(explanation_list):
    explanation = "\n".join(explanation_list)
    return chain.run(explanation=explanation)

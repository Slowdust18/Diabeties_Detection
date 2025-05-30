import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key
)

def talk_to_bot(user_message):
    response = llm.invoke([HumanMessage(content=user_message)])
    return response.content

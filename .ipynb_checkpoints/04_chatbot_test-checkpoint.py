from google import genai

client = genai.Client(api_key="AIzaSyAoj6Z8jzJ71wb1Qis5F7hU-nvKHjBDlv0")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)
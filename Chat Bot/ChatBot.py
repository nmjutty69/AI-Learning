from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, os
from dotenv import load_dotenv

#Commands in Linux
# pip install -r requirements.txt
# source ChatENV/bin/activate   this will run my created environment
# uvicorn ChatBot:app --reload  this will run backend on Linux and generates api URL which can be used to test API in Postman etc...
# open index.html in browser manually OR vs code / cursor extension "Live Server"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Tour Guide Chatbot API is running"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise tour-guide assistant for tourists"},
            {"role": "developer", "content": "Keep answers short: 1 sentence, max 100 words. Prioritize top famous places, why-it’s-worth-it. End with one short next step question."},
            {"role": "user", "content": request.message}
        ],
        "temperature": 0.69
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    result = response.json()

    return {"reply": result["choices"][0]["message"]["content"]}

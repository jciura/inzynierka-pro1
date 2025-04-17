import os

from fastapi import FastAPI
from pydantic import BaseModel
import requests
app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b-instruct"

class PrompRequest(BaseModel):
    question: str
    context: str

@app.post("/ask")
async def ask(req: PrompRequest):
    prompt = f"Context: {req.context}\n\nQuestion: {req.question}\nAnswer:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return {"answer": result["response"]}
    except Exception as e:
        return {"error": str(e)}

def load_code(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    else:
        return None

@app.post("/ask_code")
async def ask_code(file_path: str, question: str):
    code = load_code(file_path)
    if not code:
        return {"error": "No code"}
    prompt_request = PrompRequest(question=question, context=code)
    return await ask(prompt_request)
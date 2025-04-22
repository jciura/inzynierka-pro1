import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from rag.retriver import similar_questions, similar_code

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b-instruct"
CODE_EMBEDDINGS = "embeddings/embedding.jsonl"
QA_EMBEDDINGS = "embeddings/qa_embedding.json"
MODEL_NAME_EMBEDDING = "all-MiniLM-L6-v2"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
qa_data = load_json(QA_EMBEDDINGS)


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


@app.post("/ask_rag_questions")
async def ask_rag(req: PrompRequest):
    matches = similar_questions(req.question, qa_data, embedding_model)
    context = "\n\n".join(f"Q: {match[1]['question']}\nA: {match[1]['answer']}" for match in matches)
    prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return {
            "answer": result["response"],
            "used_context": context
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask_rag_code")
async def ask_code_rag(req: PrompRequest):
    code_matches = similar_code(req.question, CODE_EMBEDDINGS, embedding_model)
    context = "\n\n".join([m[1]["content"] for m in code_matches[:3]])
    prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return {
            "answer": result["response"],
            "used_context": context
        }
    except Exception as e:
        return {"error": str(e)}

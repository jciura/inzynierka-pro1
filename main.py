import os
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import httpx
from rag.retriver import similar_questions, similar_code
#from rag.generate_embeddings import generate_embeddings

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b-instruct"
# CODE_EMBEDDINGS = "embeddings/embedding.jsonl"
# QA_EMBEDDINGS = "embeddings/qa_embedding.json"
CODE_EMBEDDINGS = "embeddings/code_embedding_project_fixed.jsonl"
QA_EMBEDDINGS = "embeddings/qa_embedding_project_fixed.json"
#MODEL_NAME_EMBEDDING = "all-MiniLM-L6-v2"
CODEBERT_MODEL_NAME = "microsoft/codebert-base"

timeout = httpx.Timeout(60.0)
client = httpx.AsyncClient(timeout=timeout)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


qa_data = load_json(QA_EMBEDDINGS)


class PrompRequest(BaseModel):
    question: str
    context: str


async def response(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code,
            detail=f"Error response {exc.response.status_code} while requesting Ollama API")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred while requesting Ollama API: {exc}")

    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Unexpected error: {exc}")


@app.post("/ask")
async def ask(req: PrompRequest):
    prompt = f"Context: {req.context}\n\nQuestion: {req.question}\nAnswer:"
    answer = await response(prompt)
    return {"answer": answer}



async def load_code(file_path: str):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read file")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found")


@app.post("/ask_code")
async def ask_code(file_path: str, question: str):
    code = await load_code(file_path)
    prompt_request = PrompRequest(question=question, context=code)
    return await ask(prompt_request)


@app.post("/ask_rag_questions")
async def ask_rag(req: PrompRequest):
    try:
        matches = similar_questions(req.question, qa_data, model_name=CODEBERT_MODEL_NAME)
        context = "\n\n".join(f"Q: {m[1]['question']}\nA: {m[1]['answer']}" for m in matches)
        print(context)
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer = await response(prompt)
        print(answer)
        return {
            "answer": answer,
            "used_context": context}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in RAG: {str(exc)}")


@app.post("/ask_rag_code")
async def ask_code_rag(req: PrompRequest):
    try:
        matches = similar_code(req.question, CODE_EMBEDDINGS, model_name=CODEBERT_MODEL_NAME)
        context = "\n\n".join(m[1]["content"] for m in matches)
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer = await response(prompt)
        return {
            "answer": answer,
            "used_context": context
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in RAG: {str(exc)}")

import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from rag.retriver import similar_questions, similar_code, similar_node

#from rag.generate_embeddings import generate_embeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b-instruct"
# CODE_EMBEDDINGS = "embeddings/embedding.jsonl"
# QA_EMBEDDINGS = "embeddings/qa_embedding.json"
CODE_EMBEDDINGS = "embeddings/embedding_project_fixed.jsonl"
QA_EMBEDDINGS = "embeddings/qa_embedding_project_fixed.json"
NODE_EMBEDDINGS = "embeddings/node_embedding.json"
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
        raise HTTPException(detail=f"An error occurred while requesting Ollama API: {exc}")

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
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer = await response(prompt)
        return {
            "answer": answer,
            "used_context": context}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in RAG: {str(exc)}")


@app.post("/ask_rag_code")
async def ask_code_rag(req: PrompRequest):
    try:
        logger.info(f"Received question: {req.question}")
        logger.info(f"Using embedding file: {CODE_EMBEDDINGS}")

        if req.context:
            logger.warning("Context provided in request will be ignored  - RAG generates its own context")


        matches = similar_code(req.question, CODE_EMBEDDINGS, model_name=CODEBERT_MODEL_NAME)
        logger.info(f"Found {len(matches)} releveant code matches")
        context_parts = []
        sources = []
        for score, chunk in matches:
            file_name = chunk["metadata"]["file"]
            logger.info(f"Using code from {file_name} with similarity score: {score}")
            context_parts.append(f"# From file: {file_name}\n{chunk['content']}")
            sources.append({
                "file": file_name,
                "similarity_score": round(float(score), 3)
            })
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context length: {len(context)} character")
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer = await response(prompt)
        return {
            "answer": answer,
            "used_context": context,
            "sources": sources
        }
    except Exception as exc:
        logger.error(f"Error in RAG: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in RAG: {str(exc)}")


@app.post("/ask_rag_node")
async def ask_rag_node(req: PrompRequest):
    try:
        matches = similar_node(req.question, NODE_EMBEDDINGS, model_name=CODEBERT_MODEL_NAME)
        context = "\n\n".join(m[1]["node"] + ":\n" + m[1].get("content", "") for m in matches)
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        answer = await response(prompt)
        return {
            "answer": answer,
            "used_context": context
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in RAG Node: {str(exc)}")


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()

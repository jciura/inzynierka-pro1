from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


def similar_questions(question, qa_data, model):
    embedding = [model.encode(question)]
    results = []
    for qa in qa_data:
        qa_embedding = [model.encode(qa["question"])]
        score = cosine_similarity(embedding, qa_embedding)[0][0]
        results.append((score, qa))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:1]


def similar_code(question, code_path, model):
    with open(code_path, "r") as f:
        code_data = [json.loads(line) for line in f]
    embedding = np.array([model.encode(question)])
    results = []
    for chunk in code_data:
        chunk_embedding = np.array([chunk["embedding"]])
        score = cosine_similarity(embedding, chunk_embedding)[0][0]
        results.append((score, chunk))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]
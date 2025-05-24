from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from rag.generate_embeddings import generate_embeddings


def similar_questions(question, qa_data, model_name="microsoft/codebert-base"):
    question_embedding = generate_embeddings([question], model_name)[0]
    results = []
    print(question_embedding)
    for qa in qa_data:
        qa_embedding = np.array(qa["embedding"])
        score = cosine_similarity([question_embedding], [qa_embedding])[0][0]
        results.append((score, qa))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]


def similar_code(question, code_path, model_name="microsoft/codebert-base"):
    with open(code_path, "r") as f:
        code_data = [json.loads(line) for line in f]
    question_embedding = generate_embeddings([question], model_name)[0]
    results = []
    for chunk in code_data:
        chunk_embedding = np.array(chunk["embedding"])
        score = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
        results.append((score, chunk))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:7]
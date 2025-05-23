import json
import logging as logger
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graph.generate_embeddings_graph import generate_embeddings_graph
from rag.generate_embeddings import generate_embeddings


def extract_file_name_from_question(question):
    file_patterns = re.findall(r'\b[\w-]+\.[\w]+\b', question.lower())
    words = [word.lower() for word in question.split()]
    return file_patterns, words


def calcuate_file_name_similiarity(file_name, target_patterns, target_words):
    file_name = file_name.lower()

    file_base = file_name.rsplit('.', 1)[0]
    if file_name in target_patterns:
        return 2.0
    if file_base in target_words:
        return 1.5

    for pattern in target_patterns:
        if pattern in file_name:
            return 1.3
    return 1.0


def similar_questions(question, qa_data, model_name="microsoft/codebert-base"):
    question_embedding = generate_embeddings([question], model_name)[0]
    results = []
    for qa in qa_data:
        qa_embedding = np.array(qa["embedding"])
        score = cosine_similarity([question_embedding], [qa_embedding])[0][0]
        results.append((score, qa))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]


def similar_code(question, code_path, model_name="microsoft/codebert-base", min_similarity=0.5):
    target_patterns, target_words = extract_file_name_from_question(question)
    logger.info(f"Extracted file patterns: {target_patterns}")
    logger.info(f"Extracted words that might be files: {target_words}")

    with open(code_path, "r") as f:
        code_data = [json.loads(line) for line in f]

    question_embedding = generate_embeddings([question], model_name)[0]
    results = []

    for chunk in code_data:
        chunk_embedding = np.array(chunk["embedding"])
        base_score = cosine_similarity([question_embedding], [chunk_embedding])[0][0]

        file_name = chunk['metadata']['file'].lower()
        file_muptiplier = calcuate_file_name_similiarity(file_name, target_patterns, target_words)
        final_score = base_score * file_muptiplier
        if file_muptiplier > 1.0:
            logger.info(
                f"File {file_name} score boosted from {base_score} to {final_score} (multiplier: {file_muptiplier})")

        if final_score >= min_similarity:
            results.append((final_score, chunk))
            logger.info(f"Added chunk from {file_name} with final score {final_score}")
        else:
            logger.debug(f"Skipped chunk from {file_name} - score {final_score} below threshold {min_similarity}")

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:6]


def similar_node(question, node_embedding_path, model_name="microsoft/codebert-base"):
    with open(node_embedding_path, "r") as f:
        node_data = json.load(f)
    question_embedding = generate_embeddings_graph([question], model_name)[0]
    results = []
    for node in node_data:
        node_emb = np.array(node["embedding"])
        score = cosine_similarity([question_embedding], [node_emb])[0][0]
        importance_combined = node.get("importance", {}).get("combined", 1.0)
        final_score = score * importance_combined
        results.append((final_score, node))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:7]
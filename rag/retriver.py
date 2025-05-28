import json
import logging as logger
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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


def normalize(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

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


def extract_key_value_pairs_simple(question):
    key_terms = {"class", "method", "function", "variable", "property"}
    words = question.lower().split()
    pairs = []

    for i, word in enumerate(words):
        if word in key_terms:
            # Na razie zakładam że przed słowem kluczowym będzie jego nazwa
            pairs.append((word, words[i - 1]))

    if not pairs:
        pairs = [(None, w) for w in words if w not in key_terms and len(w) > 2]

    return pairs


def filter_repeated_codes(old_context, codes):
    filtered_codes = []
    for code in codes:
        if code not in old_context:
            filtered_codes.append(code)
    return filtered_codes


def similar_node(question, node_embedding_path, node_history_path, model_name="microsoft/codebert-base", top_k=7):
    with open(node_embedding_path, "r", encoding="utf-8") as f:
        node_data = json.load(f)

    id_to_node = {node["node"]: node for node in node_data}

    # Wyciagam pary nazwa i typ (np. User Class) i dla każdej pary robię osobno embedding, żeby wyciągnąć dobry kod
    pairs = extract_key_value_pairs_simple(question)

    embeddings_input = []
    for key, value in pairs:
        if key is not None:
            embeddings_input.append(f"{key} {value}")
        else:
            embeddings_input.append(value)

    print(embeddings_input)
    embeddings_raw = generate_embeddings_graph(embeddings_input, model_name)
    embeddings = normalize(embeddings_raw)

    results = []
    for node in node_data:
        node_emb_raw = np.array(node["embedding"])
        node_emb = normalize([node_emb_raw])[0]
        score = [cosine_similarity([emb], [node_emb])[0][0] for emb in embeddings]
        # Mnożenie wyniki przez combined importance pogorszyło wynik bo zwraca jakieś losowe węzły, które są uznane za ważne;
        final_score = max(score)
        results.append((final_score, node))

    results.sort(reverse=True, key=lambda x: x[0])
    top_nodes = results[:len(embeddings_input)]
    top_k_codes = [node["code"] for _, node in top_nodes]

    print("Top nodes:")
    for _, nod in top_nodes:
        print(nod["node"])

    all_neighbors_ids = set()
    for _, node in top_nodes:
        neigbors = node.get("related_entities", [])
        all_neighbors_ids.update(neigbors)

    sorted_neighbors = sorted(
        all_neighbors_ids,
        key=lambda nid: id_to_node.get(nid, {}).get("importance", {}).get("combined", 0.0),
        reverse=True
    )

    print("Neigbours:")
    for neighbor in sorted_neighbors:
        print(neighbor)

    neighbor_codes = [id_to_node[nid]["code"] for nid in sorted_neighbors if nid in id_to_node][:5]

    all_codes = top_k_codes + [code for code in neighbor_codes if code not in top_k_codes]

    with open(node_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        last_n = 3
        recent_history = history[-last_n:] if len(history) >= last_n else history
        old_context_parts = []
        old_context_codes = []
        for h in recent_history:
            old_context_parts.append(
                f"Q: {h.get('question', '')}\nContext:\n{h.get('context', '')}\nA: {h.get('answer', '')}\n")
            old_context_codes.append(h.get('context', ''))
        old_context = "\n---\n".join(old_context_parts)
        old_context_code = "\n\n".join(old_context_codes)

    filtered_codes = filter_repeated_codes(old_context_code, all_codes)

    joined_codes = "\n\n".join(filtered_codes)
    full_context = f"{old_context}\n\n---\n\n{joined_codes}"

    return top_nodes, full_context

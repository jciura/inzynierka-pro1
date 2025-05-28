import json
import re
from collections import defaultdict

import torch
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel

from graph.load_graph import load_gdf, extract_scores


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings_graph(texts, MODEL_NAME, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
            batch_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.extend(batch_embeddings)

    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def extract_code_block_from_file(uri, location):
    try:
        start, _ = location.split(';')  # Ignorujemy dokładny koniec, bo chcemy rozszerzyć do końca bloku
        start_line, start_col = map(int, start.split(':'))

        with open(uri, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line} >"

        block_lines = []
        open_braces = 0
        started = False

        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            block_lines.append(line.rstrip())

            # Liczymy nawiasy
            open_braces += line.count('{')
            open_braces -= line.count('}')

            # Jeśli zaczęliśmy i liczba nawiasów wróciła do 0, to koniec bloku
            if not started and open_braces > 0:
                started = True
            elif started and open_braces == 0:
                break

        code = ' '.join(block_lines)
        code = re.sub(r'\s+', ' ', code)  # Jedna spacja między wszystkimi elementami
        return code.strip()

    except Exception as e:
        return f"<Could not extract code block: {e}>"


def extract_code_from_file(uri, location):
    try:
        start, end = location.split(';')
        start_line, start_col = map(int, start.split(':'))

        with open(uri, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"<Invalid start line: {start_line} >"

        current_line = start_line - 1
        code_lines = []

        while current_line < len(lines):
            line = lines[current_line]
            code_lines.append(line.rstrip())
            current_line += 1

        code = ' '.join(code_lines)
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    except Exception as e:
        return f"<Could not extract code: {e}>"

def node_to_text(data):
    label = data.get('label', '')
    kind = data.get('kind', '')
    uri = data.get('uri', '')
    location = data.get('location', '')
    if kind in ['CLASS', 'METHOD']:
        code = extract_code_block_from_file(uri, location)
    else:
        code = extract_code_from_file(uri, location)

    return {
        "text": f"{kind} {label}",
        "kind": kind,
        "label": label,
        "code": code
    }

if __name__ == "__main__":
    MODEL_NAME = "microsoft/codebert-base"
    scg = load_gdf('../projects/scgTest.gdf')
    ccn = load_gdf('../projects/ccnTest.gdf')
    importance_scores = extract_scores("../projects/partition.js")

    reverse_ccn_map = defaultdict(list)
    for node_id in ccn.nodes():
        for neighbor in ccn.neighbors(node_id):
            reverse_ccn_map[neighbor].append(node_id)


    nodes_info = []
    texts_for_embedding = []

    for node_id, data in scg.nodes(data=True):
        node_text = node_to_text(data)
        nodes_info.append({
            "node_id": node_id,
            "kind": node_text["kind"],
            "label": node_text["label"],
            "code": node_text["code"]
        })
        texts_for_embedding.append(
            node_text["text"].lower())  # lower() żeby embeddingi były podobne dla pytań z rożnymi wielkościami liter


    embeddings = generate_embeddings_graph(texts_for_embedding, MODEL_NAME)

    json_data = []
    for info, emb in zip(nodes_info, embeddings):
        node_id = info["node_id"]
        #
        scg_neighbors = set(scg.neighbors(node_id)) if scg.has_node(node_id) else set()
        # Z grafu ccn pobieramy klasy które używaja danej klasy żeby dodać je do related_entities
        used_by = set(reverse_ccn_map[node_id]) if node_id in reverse_ccn_map else set()

        # Dla każdej metody z danej klasy pobieramy klasy w jakich dana klasa jest używana, bo może tam być użyta ta metoda
        extra_related = set()
        if info["kind"] == "METHOD":
            class_id = node_id.split('(')[0].rsplit('.', 1)[0]
            if class_id in reverse_ccn_map:
                extra_related.update(reverse_ccn_map[class_id])

        related_entities = sorted(
            scg_neighbors.union(used_by).union(extra_related),
            key=lambda nid: importance_scores["combined"].get(nid, 0.0),
            # sortowanie po loc, żeby przy wyciąganiu related_entities wybierać klasy a nie metody z przetwarzanej klasy
            reverse=True
        )

        for nid in related_entities:
            print(nid, importance_scores["combined"].get(nid, 0.0))

        json_data.append({
            "node": node_id,
            "kind": info["kind"],
            "label": info["label"],
            "related_entities": related_entities,
            "importance": {
                "loc": importance_scores['loc'].get(node_id, 0.0),
                "out-degree": importance_scores['out-degree'].get(node_id, 0.0),
                "in-degree": importance_scores['in-degree'].get(node_id, 0.0),
                "pagerank": importance_scores['pagerank'].get(node_id, 0.0),
                "eigenvector": importance_scores['eigenvector'].get(node_id, 0.0),
                "katz": importance_scores['Katz'].get(node_id, 0.0),
                "combined": importance_scores['combined'].get(node_id, 0.0),
            },
            "code": info["code"],
            "embedding": emb.tolist()
        })

    with open("../embeddings/node_embedding.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

import json

import torch
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
    return embeddings


def node_to_text(data):
    label = data.get('label', '')
    kind = data.get('kind', '')
    uri = data.get('uri', '')
    return f"{kind} {label} defined in {uri}"


if __name__ == "__main__":
    MODEL_NAME = "microsoft/codebert-base"

    g = load_gdf('../projects/test.gdf')
    documents = []
    node_ids = []
    for node, data in g.nodes(data=True):
        text = node_to_text(data)
        documents.append(text)
        node_ids.append(node)

    importance_scores = extract_scores("../projects/partition.js")
    embeddings = generate_embeddings_graph(documents, MODEL_NAME)

    json_data = []
    for node_id, emb in zip(node_ids, embeddings):
        json_data.append({
            "node": node_id,
            "embedding": emb.tolist(),
            "importance": {
                "loc": importance_scores['loc'].get(node_id, 0.0),
                "out-degree": importance_scores['out-degree'].get(node_id, 0.0),
                "in-degree": importance_scores['in-degree'].get(node_id, 0.0),
                "pagerank": importance_scores['pagerank'].get(node_id, 0.0),
                "eigenvector": importance_scores['eigenvector'].get(node_id, 0.0),
                "katz": importance_scores['Katz'].get(node_id, 0.0),
                "combined": importance_scores['combined'].get(node_id, 0.0),
            }
        })

    with open("../embeddings/node_embedding.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

import json
from sentence_transformers import SentenceTransformer


def code_embeddings(input_path, output_path, model_name):
    model = SentenceTransformer(model_name)
    with open(input_path, "r") as f:
        chunks = json.load(f)
    with open(output_path, "w") as out_f:
        for chunk in chunks:
            embedding = model.encode(chunk["content"]).tolist()
            chunk_data = {
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": {
                    "file": chunk.get("file"),
                    "type": chunk.get("type"),
                    "class": chunk.get("class"),
                    "method": chunk.get("method")
                }
            }
            out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")


def questions_embeddings(qa_data, output_path, model_name):
    model = SentenceTransformer(model_name)
    #data = load_qa_data(qa_data)
    for qa in qa_data:
        question = qa["question"]
        embedding = model.encode(question).tolist()
        qa["embedding"] = embedding
    with open(output_path, "w") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)


import json
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(texts, model_name="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    batch_size = 2
    embeddings = []
    n = len(texts)
    for i in range(0, n, batch_size):
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


def code_embeddings(input_path, output_path, model_name="microsoft/codebert-base"):
    with open(input_path, "r") as f:
        chunks = json.load(f)
    chunk_contents = []
    for chunk in chunks:
        chunk_contents.append(chunk["content"])
    embeddings = generate_embeddings(chunk_contents, model_name)
    with open(output_path, "w") as out_f:
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk["content"],
                "embedding": embeddings[i].tolist(),
                "metadata": {
                    "file": chunk.get("file"),
                    "type": chunk.get("type"),
                    "class": chunk.get("class"),
                    "method": chunk.get("method")
                }
            }
            out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")


def questions_embeddings(qa_data, output_path, model_name="microsoft/codebert-base"):
    questions = []
    for qa in qa_data:
        questions.append(qa["question"])
    all_embeddings = generate_embeddings(questions, model_name)
    for i, qa in enumerate(qa_data):
        qa["embedding"] = all_embeddings[i].tolist()
    with open(output_path, "w") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    input_path = "../data/project_chunks_fixed.json"
    output_path = "../embeddings/embedding_project_fixed.jsonl"


    code_embeddings(input_path, output_path)

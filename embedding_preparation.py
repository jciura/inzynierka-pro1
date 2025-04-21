from rag.chunking import get_all_chunks
from rag.generate_embeddings import code_embeddings, questions_embeddings
import json


CODE_DIR = "code"
QA_INPUT = "data/q&a.json"
CHUNKS_JSON = "data/chunks.json"
CODE_EMBEDDINGS = "embeddings/embedding.jsonl"
QA_EMBEDDINGS = "embeddings/qa_embedding.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


get_all_chunks(CODE_DIR, CHUNKS_JSON)
code_embeddings(CHUNKS_JSON, CODE_EMBEDDINGS, MODEL_NAME)
m = load_json(QA_INPUT)
questions_embeddings(m, QA_EMBEDDINGS, MODEL_NAME)




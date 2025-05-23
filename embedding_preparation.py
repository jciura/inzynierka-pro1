from rag.chunking import get_all_chunks
from rag.generate_embeddings import code_embeddings, questions_embeddings
import json


CODE_DIR = "projects/rich"
QA_INPUT = "data/q&a_project.json"
CHUNKS_JSON = "data/project_chunks_fixed.json"
CODE_EMBEDDINGS = "embeddings/embedding_project.jsonl"
QA_EMBEDDINGS = "embeddings/qa_embedding_project_fixed.json"
MODEL_NAME = "microsoft/codebert-base"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# get_all_chunks(CODE_DIR, CHUNKS_JSON)
# code_embeddings(CHUNKS_JSON, CODE_EMBEDDINGS, MODEL_NAME)
m = load_json(QA_INPUT)
questions_embeddings(m, QA_EMBEDDINGS, MODEL_NAME)




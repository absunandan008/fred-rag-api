from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PATH

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()
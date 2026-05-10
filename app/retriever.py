import chromadb
from app.config import CHROMA_PATH
from app.embedder import get_embedding

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="fred")


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for i, doc in enumerate(results["documents"][0]):
        docs.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return docs
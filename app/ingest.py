import requests
from app.config import FRED_API_KEY, CHROMA_PATH
from app.embedder import get_embeddings

import chromadb

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "UNRATE": "unemployment rate",
    "CPIAUCSL": "consumer price index (CPI)",
    "FEDFUNDS": "federal funds rate",
    "GDP": "gross domestic product (GDP)",
    "GDPC1": "real GDP",
    "PCE": "personal consumption expenditures",
    "PAYEMS": "total nonfarm payrolls",
    "T10YIR": "10-year treasury inflation-indexed security rate",
}

def fetch_series(series_id: str) -> list[dict]:
    response = requests.get(FRED_BASE_URL, params={
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 60
    })
    response.raise_for_status()
    return response.json()["observations"]

def build_chunks(series_id: str, label: str, observations: list[dict]) -> list[str]:
    chunks = []
    for obs in observations:
        if obs["value"] == ".":
            continue
        text = f"In {obs['date']}, the {label} ({series_id}) was {obs['value']}."
        chunks.append(text)
    return chunks

def ingest():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="fred")

    for series_id, label in SERIES.items():
        print(f"Fetching {series_id}...")
        observations = fetch_series(series_id)
        chunks = build_chunks(series_id, label, observations)
        embeddings = get_embeddings(chunks)

        ids = [f"{series_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"series_id": series_id, "label": label} for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"  ✓ {len(chunks)} chunks ingested for {series_id}")

    print("Ingest complete.")

if __name__ == "__main__":
    ingest()
        

        

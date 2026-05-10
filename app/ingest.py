import requests
from app.config import FRED_API_KEY, CHROMA_PATH
from app.embedder import get_embeddings

import chromadb

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "UNRATE": "unemployment rate",
    "CPIAUCSL": "consumer price index (CPI)",
    "GDP": "gross domestic product (GDP)",
    "GDPC1": "real GDP",
    "PCE": "personal consumption expenditures",
    "PAYEMS": "total nonfarm payrolls",
    "DGS10": "10-year treasury yield",
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

def build_chunks(series_id: str, label: str, observations: list[dict]) -> tuple[list[str], list[dict]]:
    chunks = []
    metadatas = []
    window_size = 6

    observations = [o for o in observations if o["value"] != "."]

    for i in range(0, len(observations), window_size):
        window = observations[i:i + window_size]
        if not window:
            continue

        lines = "\n".join([f"  {o['date']}: {o['value']}" for o in window])
        text = f"The {label} ({series_id}) from {window[-1]['date']} to {window[0]['date']}:\n{lines}"

        chunks.append(text)
        metadatas.append({
            "series_id": series_id,
            "label": label,
            "date_from": window[-1]["date"],
            "date_to": window[0]["date"],
            "date_to_int": int(window[0]["date"].replace("-", "")),
        })

    return chunks, metadatas

def ingest():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="fred")

    for series_id, label in SERIES.items():
        print(f"Fetching {series_id}...")
        observations = fetch_series(series_id)
        chunks, metadatas = build_chunks(series_id, label, observations)
        embeddings = get_embeddings(chunks)

        ids = [f"{series_id}_{i}" for i in range(len(chunks))]
        #metadatas = [{"series_id": series_id, "label": label} for _ in chunks]

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
        

        

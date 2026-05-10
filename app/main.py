from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.retriever import retrieve
from app.llm import get_llm_response
from app.config import LLM_PROVIDER, OLLAMA_MODEL, BEDROCK_MODEL_ID

app = FastAPI(title="FRED RAG API")


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    date_from: str = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    context = retrieve(request.question, n_results=request.n_results, date_from=request.date_from)

    if not context:
        raise HTTPException(status_code=404, detail="No relevant context found")

    context_text = "\n".join([doc["text"] for doc in context])

    prompt = f"""You are a macroeconomic data assistant. Answer the question using only the context provided below.
If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {request.question}
Answer:"""

    answer = get_llm_response(prompt)
    model = BEDROCK_MODEL_ID if LLM_PROVIDER == "bedrock" else OLLAMA_MODEL

    return QueryResponse(
        question=request.question,
        answer=answer,
        context=context,
        model=model,
    )
import boto3
import ollama
from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, BEDROCK_MODEL_ID, AWS_REGION

def get_llm_response(prompt: str) -> str:
    if LLM_PROVIDER == "bedrock":
        return _bedrock_response(prompt)
    return _ollama_response(prompt)

def _ollama_response(prompt: str) -> str:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content

def _bedrock_response(prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}]
    )
    return response["output"]["message"]["content"][0]["text"]
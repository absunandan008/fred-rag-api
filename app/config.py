from dotenv import load_dotenv
import os

load_dotenv()

#LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

#ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Bedrock
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ChromaDB
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore")

# FRED
FRED_API_KEY = os.getenv("FRED_API_KEY")
"""
LLM Provider Factory for the Legal Contract Reviewer.
"""
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_SMALL_MODEL = "llama3.1:8b"
OLLAMA_LARGE_MODEL = "qwen2.5:7b"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


def get_llm(
    temperature: float = 0.0,
    prefer_large: bool = False,
    provider: str = "ollama",
):
    """Return the requested chat model."""
    provider = provider.lower()

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    model = OLLAMA_LARGE_MODEL if prefer_large else OLLAMA_SMALL_MODEL
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def get_embedding_model(provider: str = "ollama"):
    """Return the embedding model for the legal RAG index."""
    if provider.lower() == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    try:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    except Exception:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

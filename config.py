import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    """LLM, embedding, and cross-encoder service configuration."""
    api_key: str
    base_url: str
    model: str
    rerank_url: str
    rerank_api_key: str
    rerank_model: str
    embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_env(cls, prefix="GPT_4o_mini"):
        """Load only the services used by SF-RAG."""
        load_dotenv()
        api_key = os.getenv(f"{prefix}.api_key")
        base_url = os.getenv(f"{prefix}.base_url")
        model = os.getenv(f"{prefix}.model")
        rerank_url = os.getenv(f"{prefix}.rerank_url")
        rerank_api_key = os.getenv(f"{prefix}.rerank_api_key")
        rerank_model = os.getenv(f"{prefix}.rerank_model")

        if not all([api_key, base_url, model, rerank_url, rerank_api_key, rerank_model]):
            raise ValueError(f"Missing SF-RAG LLM/reranker variables under {prefix}")
        embedding_model = os.getenv(f"{prefix}.embedding_model", "text-embedding-3-small")
        return cls(api_key, base_url, model, rerank_url, rerank_api_key,
                   rerank_model, embedding_model)



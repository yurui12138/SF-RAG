import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration management class"""
    api_key: str
    base_url: str
    model: str
    model_vl: str
    rerank_url: str
    rerank_api_key: str
    rerank_model: str
    picture_bed_url: str
    picture_bed_token: str

    @classmethod
    def from_env(cls, prefix=""):
        """Load configuration from environment variables"""
        load_dotenv()
        api_key = os.getenv(f"{prefix}.api_key")
        base_url = os.getenv(f"{prefix}.base_url")
        model = os.getenv(f"{prefix}.model")
        model_vl = os.getenv(f"{prefix}.model_vl")
        rerank_url = os.getenv(f"{prefix}.rerank_url")
        rerank_api_key = os.getenv(f"{prefix}.rerank_api_key")
        rerank_model = os.getenv(f"{prefix}.rerank_model")
        picture_bed_url = os.getenv(f"{prefix}.picture_bed_url")
        picture_bed_token = os.getenv(f"{prefix}.picture_bed_token")

        if not all([api_key, base_url, model, model_vl, rerank_url, rerank_api_key, rerank_model, picture_bed_url, picture_bed_token]):
            raise ValueError(f"Missing environment variables: {prefix}.api_key, {prefix}.base_url, {prefix}.model, {prefix}.model_vl, {prefix}.rerank_url, {prefix}.rerank_api_key, {prefix}.rerank_model, {prefix}.picture_bed_url, {prefix}.picture_bed_token")
        return cls(api_key, base_url, model, model_vl, rerank_url, rerank_api_key, rerank_model, picture_bed_url, picture_bed_token)



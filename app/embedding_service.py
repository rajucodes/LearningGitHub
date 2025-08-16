import os
from typing import List, Optional

# Fast fallback
from fastembed import TextEmbedding

_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = _OPENAI_AVAILABLE and bool(self.openai_api_key)
        self.model_name = model_name or ("text-embedding-3-small" if self.use_openai else "BAAI/bge-small-en-v1.5")

        if self.use_openai:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.dimension: Optional[int] = 1536  # text-embedding-3-small
        else:
            self.client = TextEmbedding(self.model_name)
            # Initialize once to get dimension
            vector = next(self.client.embed(["dimension probe"]))
            self.dimension = len(vector)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            # OpenAI batched embeddings
            resp = self.client.embeddings.create(model=self.model_name, input=texts)
            return [d.embedding for d in resp.data]
        else:
            return [list(vec) for vec in self.client.embed(texts)]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
"""Batched embeddings via Nebius Token Factory's OpenAI-compatible API.

Why this exists
---------------
`langchain-nebius`' `NebiusEmbeddings.embed_documents` was issuing one request
per document, which hit Nebius rate limits and triggered exponential-backoff
retries — embedding ~30 chunks took ~12 minutes and timed the notebook out.

Sending all the texts in a single batched request to the **same** Nebius endpoint
(`api.tokenfactory.nebius.com`, OpenAI-compatible) does the same work in ~6 seconds.
This thin wrapper does exactly that and plugs into LangChain anywhere an
`Embeddings` object is expected (PineconeVectorStore, retrievers, etc.).

It is still a Nebius Token Factory model call — it uses the Nebius base URL, the
`NEBIUS_API_KEY`, and the Nebius-hosted `Qwen/Qwen3-Embedding-8B` model.
"""

import os

from langchain_core.embeddings import Embeddings
from openai import OpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"  # 4096-dim


class NebiusBatchEmbeddings(Embeddings):
    """LangChain Embeddings backed by Nebius Token Factory, batched per request."""

    def __init__(self, model=DEFAULT_MODEL, api_key=None,
                 base_url=NEBIUS_BASE_URL, batch_size=100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ["NEBIUS_API_KEY"],
        )

    def embed_documents(self, texts):
        """Embed many texts, sending up to `batch_size` per request (one request here)."""
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts[start:start + self.batch_size],
            )
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text):
        """Embed a single query string."""
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding

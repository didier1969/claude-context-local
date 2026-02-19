"""Ollama embedding model implementation — calls a local Ollama server.

Standalone class (no torch dependency) that implements the same interface
as EmbeddingModel but communicates via HTTP with an Ollama instance.
"""

import os
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional

# nomic-embed-text context window is 8192 tokens (~24k chars conservatively)
MAX_CHARS_PER_INPUT = 20000


class OllamaEmbeddingModel:
    """Embedding model that delegates to a running Ollama instance."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.base_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:41434"
        )
        self._embedding_dim: Optional[int] = None
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"OllamaEmbeddingModel initialized: model={model_name}, url={self.base_url}"
        )

    # -- core interface --------------------------------------------------------

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts via Ollama /api/embed endpoint (batch).

        Truncates inputs exceeding MAX_CHARS_PER_INPUT to stay within
        the model context window.
        """
        if not texts:
            return np.array([])

        # Truncate oversized inputs
        safe_texts = [
            t[:MAX_CHARS_PER_INPUT] if len(t) > MAX_CHARS_PER_INPUT else t
            for t in texts
        ]

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model_name, "input": safe_texts}

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        embeddings = np.array(data["embeddings"], dtype=np.float32)

        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        return embeddings

    def get_embedding_dimension(self) -> int:
        if self._embedding_dim is not None:
            return self._embedding_dim
        self.encode(["hello"])
        return self._embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "ollama",
            "base_url": self.base_url,
            "embedding_dimension": self._embedding_dim or "unknown (call encode first)",
            "status": "loaded",
        }

    def cleanup(self):
        pass  # stateless HTTP client

    # Mimic SentenceTransformer `.model` property for compatibility
    @property
    def model(self):
        return self

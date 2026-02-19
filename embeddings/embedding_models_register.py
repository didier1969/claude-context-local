"""Embedding models registry."""
from embeddings.ollama import OllamaEmbeddingModel

AVAILIABLE_MODELS = {
    "ollama/nomic-embed-text": OllamaEmbeddingModel,
}

# Gemma requires sentence_transformers — register only if available
try:
    from embeddings.gemma import GemmaEmbeddingModel
    AVAILIABLE_MODELS["google/embeddinggemma-300m"] = GemmaEmbeddingModel
except ImportError:
    pass

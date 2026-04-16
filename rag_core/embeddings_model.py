from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import EMBEDDING_MODEL

load_dotenv()


def get_embeddings() -> HuggingFaceEmbedding:
    """Return the shared embedding model used for indexing and retrieval."""
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        normalize=True,
    )

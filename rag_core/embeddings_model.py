# rag_core/embeddings_model.py
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

load_dotenv()  # loads HUGGINGFACEHUB_API_TOKEN if present

def get_embeddings():
    """
    Returns a HuggingFaceEmbeddings instance using all-MiniLM.
    This runs locally via sentence-transformers and will download
    the model from Hugging Face (using your HF token if needed).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},      # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # optional but often helpful
    )

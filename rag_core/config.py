from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

VECTORSTORE_PATH = str(VECTORSTORE_DIR / "career_faiss_index")

GROQ_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_EVAL_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

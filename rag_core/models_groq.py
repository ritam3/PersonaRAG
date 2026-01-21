# rag_core/models_groq.py
from dotenv import load_dotenv
from langchain_core.callbacks import Callbacks
from langchain_core.caches import BaseCache
from langchain_groq import ChatGroq
from .config import GROQ_CHAT_MODEL, GROQ_EVAL_MODEL

load_dotenv()
ChatGroq.model_rebuild()

def get_answer_llm():
    return ChatGroq(model=GROQ_CHAT_MODEL, temperature=0.1)

def get_judge_llm():
    return ChatGroq(model=GROQ_EVAL_MODEL, temperature=0.0)

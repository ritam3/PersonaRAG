from dotenv import load_dotenv
from llama_index.llms.groq import Groq

from .config import GROQ_CHAT_MODEL, GROQ_EVAL_MODEL

load_dotenv()


def get_answer_llm() -> Groq:
    return Groq(model=GROQ_CHAT_MODEL, temperature=0.1)


def get_judge_llm() -> Groq:
    return Groq(model=GROQ_EVAL_MODEL, temperature=0.0)

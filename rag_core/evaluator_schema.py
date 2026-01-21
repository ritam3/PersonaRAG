# rag_core/evaluator_schema.py
from langchain_core.pydantic_v1 import BaseModel, Field

class EvalResult(BaseModel):
    follows_system_prompt: bool = Field(...)
    answers_user_question: bool = Field(...)
    grounded_in_context_score: float = Field(ge=0.0, le=1.0)
    hallucination_detected: bool = Field(...)
    overall_score: float = Field(ge=0.0, le=1.0)
    should_retry: bool = Field(...)
    feedback: str = Field(...)

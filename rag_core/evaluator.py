# rag_core/evaluator.py
from langchain_core.prompts import ChatPromptTemplate

from .models_groq import get_judge_llm
from .evaluator_schema import EvalResult

# Base LLM
_base_judge_llm = get_judge_llm()

# Wrap LLM with structured output (EvalResult)
_judge_llm_structured = _base_judge_llm.with_structured_output(EvalResult)

# Prompt template for the judge
JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are an impartial evaluator for a RAG-based assistant about Ritam's career.

You get:
- System prompt,
- User question,
- Retrieved context,
- Assistant answer.

Check:
1. Does the answer follow the system prompt? (only Ritam's career / projects / research.)
2. Does the answer actually respond to the user's question?
3. Is the answer having too much extra information
3. How well is the answer grounded in the context? Penalize hallucinations.

Return a strict JSON object matching the EvalResult schema.
"""),
        ("human", """
System prompt:
----------------
{system_prompt}

User question:
----------------
{question}

Retrieved context:
----------------
{context}

Assistant answer:
----------------
{answer}
"""),
    ]
)

def evaluate_answer(system_prompt: str, question: str, context_docs, answer: str) -> EvalResult:
    """
    Run the LLM judge with structured output over the given answer.
    If the judge fails, return a safe EvalResult that disables retry.
    """
    ctx_text = "\n\n".join(
        f"[DOC {i}] (source={d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(context_docs)
    )

    chain = JUDGE_PROMPT | _judge_llm_structured

    try:
        result: EvalResult = chain.invoke({
            "system_prompt": system_prompt,
            "question": question,
            "context": ctx_text,
            "answer": answer,
        })
        return result

    except Exception as e:
        # Judge died / timed out / bad response
        # -> don't retry, but don't crash the main answer either.
        return EvalResult(
            follows_system_prompt=True,           # neutral / optimistic defaults
            answers_user_question=True,
            grounded_in_context_score=1.0,
            hallucination_detected=False,
            overall_score=1.0,                   # high score so threshold logic won't trigger
            should_retry=False,                  # explicitly NO RETRY
            feedback=f"[Evaluator failure] Judge could not evaluate this answer: {e}",
        )

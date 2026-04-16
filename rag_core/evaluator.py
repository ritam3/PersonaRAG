import json
import re

from .evaluator_schema import EvalResult
from .models_groq import get_judge_llm


_judge_llm = get_judge_llm()

JUDGE_PROMPT = """
You are an impartial evaluator for a RAG-based assistant about Ritam's career.

You get:
- System prompt
- User question
- Retrieved context
- Assistant answer

Check:
1. Does the answer follow the system prompt? (only Ritam's career / projects / research.)
2. Does the answer actually respond to the user's question?
3. Is the answer having too much extra information?
4. How well is the answer grounded in the context? Penalize hallucinations.

Return JSON only using this schema:
{{
  "follows_system_prompt": true,
  "answers_user_question": true,
  "grounded_in_context_score": 0.0,
  "hallucination_detected": false,
  "overall_score": 0.0,
  "should_retry": false,
  "feedback": "short explanation"
}}

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
""".strip()


def _response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response).strip()


def _parse_eval_json(text: str) -> dict:
    raw = (text or "").strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced_match:
        raw = fenced_match.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def evaluate_answer(system_prompt: str, question: str, context_docs, answer: str) -> EvalResult:
    """Run the judge model over the given answer with a strict JSON schema."""
    ctx_text = "\n\n".join(
        f"[DOC {i}] (source={d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(context_docs)
    )

    prompt = JUDGE_PROMPT.format(
        system_prompt=system_prompt,
        question=question,
        context=ctx_text,
        answer=answer,
    )

    try:
        result = _judge_llm.complete(prompt)
        payload = _parse_eval_json(_response_text(result))
        return EvalResult.model_validate(payload)
    except Exception as e:
        return EvalResult(
            follows_system_prompt=True,
            answers_user_question=True,
            grounded_in_context_score=1.0,
            hallucination_detected=False,
            overall_score=1.0,
            should_retry=False,
            feedback=f"[Evaluator failure] Judge could not evaluate this answer: {e}",
        )

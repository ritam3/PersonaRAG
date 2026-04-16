import re

from rag_core.models_groq import get_judge_llm


_rewriter_llm = get_judge_llm()
_MAX_HISTORY_TURNS = 4
_MAX_HISTORY_CHARS = 2500

QUESTION_REWRITE_PROMPT = """
You are a helpful assistant that rewrites follow-up questions into standalone questions.

You are given a chat history between a user and an assistant, plus the user's new question.
Your job is to rewrite the new question so that it is self-contained and can be understood
without the previous turns.

The rewritten question MUST stay faithful to the user's intent and be about Ritam's
career, projects, research, or education.
If the question is already standalone, return it as-is.

Important rules:
- Preserve the user's original wording for project names, aliases, shorthand references,
  and distinctive phrases whenever possible.
- If the user says things like "the improper face one", "that Paytm one", "the PersonaRAG project",
  or similar shorthand, keep those phrases in the rewritten question rather than replacing them
  with a paraphrased interpretation.
- Use chat history only to add missing context, not to overwrite or over-interpret the user's wording.
- Prefer minimal rewriting. Make the question self-contained, but do not add speculative details.
- If a follow-up refers to a previously mentioned item, rewrite by naming that item while preserving
  any distinctive user phrase that helps retrieval.
- Return ONLY the rewritten question inside <rewrite>...</rewrite>.
- Do not add explanations, bullets, labels, markdown, or surrounding quotes.

Good rewrite examples:
- "explain the improper face one as well"
  -> "Explain the 'improper face' project as well."
- "what about the Paytm one?"
  -> "What about the Paytm-related project?"

Bad rewrite examples:
- "explain the improper face one as well"
  -> "What project at Paytm Money involved using image processing and deep learning to alert users about potential issues with their photos?"

Chat history:
------------
{chat_history}

New user question:
------------
{question}

Rewrite the new question into a single, self-contained question.
Return ONLY:
<rewrite>your rewritten question here</rewrite>
""".strip()


def _response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response).strip()


def _compact_history(history) -> str:
    turns = []
    for turn in history[-_MAX_HISTORY_TURNS:]:
        if not turn or len(turn) < 2:
            continue
        user_msg, assistant_msg = turn[0], turn[1]
        turns.append(f"User: {user_msg}")
        turns.append(f"Assistant: {assistant_msg}")

    history_text = "\n".join(turns).strip()
    if len(history_text) > _MAX_HISTORY_CHARS:
        history_text = history_text[-_MAX_HISTORY_CHARS:]
    return history_text


def _extract_rewrite(raw_text: str, fallback_question: str) -> str:
    raw = (raw_text or "").strip()
    if not raw:
        return fallback_question

    match = re.search(r"<rewrite>\s*(.*?)\s*</rewrite>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = raw

    candidate = re.sub(r"^```(?:text)?\s*", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\s*```$", "", candidate).strip()
    candidate = re.sub(r'^(rewritten question|rewrite)\s*:\s*', "", candidate, flags=re.IGNORECASE).strip()
    candidate = candidate.strip('"').strip("'").strip()
    candidate = " ".join(candidate.split())

    if not candidate:
        return fallback_question
    if len(candidate) > 400:
        return fallback_question
    return candidate


def rewrite_question_with_history(history, question: str) -> str:
    """Rewrite a follow-up into a standalone question using the judge model."""
    if not history:
        return question

    history_text = _compact_history(history)
    prompt = QUESTION_REWRITE_PROMPT.format(
        chat_history=history_text,
        question=question,
    )
    response = _rewriter_llm.complete(prompt)
    standalone = _extract_rewrite(_response_text(response), question)
    return standalone or question

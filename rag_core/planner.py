import json
import logging
import re
from typing import Any, Dict, List, Sequence

from langchain_core.prompts import ChatPromptTemplate

from rag_core.models_groq import get_judge_llm
from rag_core.planner_schema import (
    ExtractedEntities,
    HeaderCatalogEntry,
    RetrievalPlan,
)


logger = logging.getLogger("model_flow")

_planner_llm = get_judge_llm()


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are planning retrieval steps for a RAG assistant about Ritam's career.

You will receive:
- the user question
- a catalog of REAL headers from the indexed website/resume data

Return a strict JSON object matching the RetrievalPlan schema.

Required top-level keys:
- strategy
- reasoning
- steps

Rules:
- Use the provided headers to decide which sections to search.
- Prefer 1 to 3 steps.
- Use `retrieve` when the system should search docs.
- Use `extract` when a later retrieval depends on entities from a previous step.
- If a phrase is referential or ambiguous, such as "the project", "it", or "that role",
  first retrieve the anchoring section, then add an `extract` step, then retrieve again.
- `target_headers` should use header strings from the provided catalog when possible.
- `query` should be a retrieval query, not the final answer.
- Keep the plan concise and executable.
- Return JSON only. Do not wrap it in markdown.
""".strip(),
        ),
        (
            "human",
            """
User question:
{question}

Header catalog:
{header_catalog}
""".strip(),
        ),
    ]
)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You extract entities from career-related context for retrieval follow-up.

Return a strict JSON object matching the ExtractedEntities schema.

Rules:
- Only extract entities explicitly supported by the provided context.
- Prefer exact names/titles when available.
- Leave fields empty if the context does not support them.
- Return JSON only. Do not wrap it in markdown.
""".strip(),
        ),
        (
            "human",
            """
Fields requested:
{fields}

Context:
{context}
""".strip(),
        ),
    ]
    )


def _parse_json_payload(text: str) -> Dict[str, Any]:
    """Parse a JSON object from raw LLM text without tool-calling support."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty planner response")

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


def _normalize_strategy(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "single": "single_section",
        "single_step": "single_section",
        "single_section": "single_section",
        "multi": "multi_section",
        "multi_step": "multi_section",
        "multi_section": "multi_section",
        "dependent": "dependent_lookup",
        "dependent_retrieval": "dependent_lookup",
        "dependent_lookup": "dependent_lookup",
        "sequential": "dependent_lookup",
        "stepwise": "dependent_lookup",
    }
    return mapping.get(normalized, "single_section")


def _normalize_plan_payload(payload: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Coerce near-valid planner output into our strict schema before validation."""
    normalized = dict(payload or {})
    normalized["strategy"] = _normalize_strategy(normalized.get("strategy", "single_section"))
    normalized["reasoning"] = normalized.get("reasoning") or "Planner response normalized."

    raw_steps = normalized.get("steps") or []
    steps: List[Dict[str, Any]] = []
    last_retrieve_query = question
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue

        action = (raw_step.get("action") or "retrieve").strip().lower()
        if action not in {"retrieve", "extract"}:
            action = "retrieve"

        depends_on = raw_step.get("depends_on")
        if depends_on == "":
            depends_on = None

        query = raw_step.get("query") or ""
        if action == "retrieve":
            query = query.strip() or last_retrieve_query or question
            last_retrieve_query = query

        step = {
            "step_id": raw_step.get("step_id") or f"step_{index}",
            "action": action,
            "purpose": raw_step.get("purpose") or f"Execute {action} step {index}.",
            "query": query,
            "target_headers": raw_step.get("target_headers") or [],
            "depends_on": depends_on,
            "extract_fields": raw_step.get("extract_fields") or [],
        }
        steps.append(step)

    if not steps:
        steps = [
            {
                "step_id": "step_1",
                "action": "retrieve",
                "purpose": "Retrieve context for the original user question.",
                "query": question,
                "target_headers": [],
                "depends_on": None,
                "extract_fields": [],
            }
        ]

    normalized["steps"] = steps
    return normalized


def build_header_catalog(docs: Sequence[Any]) -> List[HeaderCatalogEntry]:
    """Build a deduplicated planner catalog using title-level entries only."""
    entries: List[HeaderCatalogEntry] = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}
        if not metadata.get("planner_include", False):
            continue

        header = (metadata.get("page_title") or metadata.get("section_header") or metadata.get("section_label") or "").strip()
        if not header:
            continue

        section_label = (metadata.get("section_label") or "").strip()
        section_type = (metadata.get("section_type") or "").strip()
        source = (metadata.get("source") or "").strip()
        summary = (metadata.get("page_summary") or "").strip()
        key = (header.lower(), source.lower())
        if key in seen:
            continue

        seen.add(key)
        entries.append(
            HeaderCatalogEntry(
                header=header,
                section_label=section_label,
                section_type=section_type,
                source=source,
                summary=summary,
            )
        )

    return entries


def _serialize_header_catalog(header_catalog: Sequence[HeaderCatalogEntry]) -> str:
    payload = [
        {
            "header": entry.header,
            "section_label": entry.section_label,
            "section_type": entry.section_type,
            "source": entry.source,
            "summary": entry.summary,
        }
        for entry in header_catalog
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def fallback_plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        strategy="single_section",
        reasoning="Fallback plan after planner failure.",
        steps=[
            {
                "step_id": "step_1",
                "action": "retrieve",
                "purpose": "Retrieve context for the original user question.",
                "query": question,
                "target_headers": [],
                "depends_on": None,
                "extract_fields": [],
            }
        ],
    )


def plan_retrieval(question: str, header_catalog: Sequence[HeaderCatalogEntry]) -> RetrievalPlan:
    """Use the planner LLM to generate a constrained retrieval plan."""
    chain = PLANNER_PROMPT | _planner_llm
    try:
        response = chain.invoke(
            {
                "question": question,
                "header_catalog": _serialize_header_catalog(header_catalog),
            }
        )
        payload = _parse_json_payload(getattr(response, "content", ""))
        payload = _normalize_plan_payload(payload, question)
        plan = RetrievalPlan.parse_obj(payload)
        if not plan.steps:
            return fallback_plan(question)
        return plan
    except Exception as exc:
        logger.exception(
            "planner.error | %s",
            {"question": question, "error": str(exc)},
        )
        return fallback_plan(question)


def extract_entities_from_docs(docs: Sequence[Any], fields: Sequence[str]) -> ExtractedEntities:
    """Extract follow-up retrieval entities from retrieved documents."""
    if not docs or not fields:
        return ExtractedEntities()

    context = "\n\n".join(
        f"[DOC {idx}] source={(doc.metadata or {}).get('source', 'unknown')}\n{doc.page_content}"
        for idx, doc in enumerate(docs[:4])
    )

    chain = EXTRACTION_PROMPT | _planner_llm
    try:
        response = chain.invoke(
            {
                "fields": ", ".join(fields),
                "context": context,
            }
        )
        payload = _parse_json_payload(getattr(response, "content", ""))
        return ExtractedEntities.parse_obj(payload)
    except Exception as exc:
        logger.exception(
            "planner.extract_error | %s",
            {"fields": list(fields), "error": str(exc)},
        )
        return ExtractedEntities()


def entities_to_map(entities: ExtractedEntities) -> Dict[str, str]:
    """Flatten extracted entity lists into query-template placeholders."""
    return {
        "project_names": ", ".join(entities.project_names),
        "company_names": ", ".join(entities.company_names),
        "role_titles": ", ".join(entities.role_titles),
        "keywords": ", ".join(entities.keywords),
    }

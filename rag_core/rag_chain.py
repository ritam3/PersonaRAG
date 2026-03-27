# rag_core/rag_chain.py (top of file)

import copy
import logging
import re
from typing import List, Tuple, Dict, Any
from langchain.schema import Document, BaseRetriever
from transformers import pipeline

from rag_core.planner import (
    build_header_catalog,
    entities_to_map,
    extract_entities_from_docs,
    plan_retrieval,
)

# ---- Label routing helpers ----

# Global zero-shot classifier (loaded once)
label_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
logger = logging.getLogger("model_flow")

def build_label_vocab(docs: List[Document]) -> List[str]:
    labels = []
    seen = set()
    for d in docs:
        header = (d.metadata.get("section_header") or "").strip()
        s_label = (d.metadata.get("section_label") or "").strip()
        candidates = [header, s_label]
        for c in candidates:
            if not c:
                continue
            normalized = " ".join(c.split())
            if len(normalized) > 120:
                normalized = normalized[:120] + "..."
            if normalized not in seen:
                seen.add(normalized)
                labels.append(normalized)
    return labels

def map_query_to_labels_zero_shot(
    query: str,
    candidate_labels: List[str],
    top_k: int = 5,
    score_threshold: float = 0.40,
) -> List[Tuple[str, float]]:
    if not candidate_labels:
        return []
    out = label_classifier(query, candidate_labels, multi_label=True)
    labels_out = out["labels"]
    scores_out = out["scores"]
    selected: List[Tuple[str, float]] = []
    for lbl, score in zip(labels_out[:top_k], scores_out[:top_k]):
        if score >= score_threshold:
            selected.append((lbl, float(score)))
    if not selected and labels_out:
        selected = [(labels_out[0], float(scores_out[0]))]
    return selected

def fetch_docs_by_labels_with_scores(
    selected_labels: List[Tuple[str, float]],
    docs: List[Document],
) -> List[Tuple[Document, float, List[str]]]:
    """
    For each doc, determine which of selected_labels it matches
    (substring match on section_header / section_label).
    Return (Document, score, [matched_labels]) where score is max label score.
    """
    if not selected_labels:
        return []
    label_score: Dict[str, float] = {lbl.lower(): sc for lbl, sc in selected_labels}
    out: List[Tuple[Document, float, List[str]]] = []
    for d in docs:
        header = (d.metadata.get("section_header") or "").lower()
        s_label = (d.metadata.get("section_label") or "").lower()
        combined = header + " " + s_label

        matched_labels: List[str] = []
        matched_scores: List[float] = []
        for lbl_lower, sc in label_score.items():
            if lbl_lower and lbl_lower in combined:
                matched_labels.append(lbl_lower)
                matched_scores.append(sc)

        if matched_labels:
            doc_score = max(matched_scores)
            out.append((d, doc_score, matched_labels))
    return out


def _doc_key(doc: Document) -> Tuple[str, str]:
    return (doc.metadata.get("source"), doc.page_content[:200])


def _normalize_header_name(value: str) -> str:
    value = value or ""
    value = value.lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _doc_matches_headers(doc: Document, target_headers: List[str]) -> bool:
    if not target_headers:
        return True

    metadata = doc.metadata or {}
    doc_headers = [
        _normalize_header_name(metadata.get("section_header", "")),
        _normalize_header_name(metadata.get("section_label", "")),
    ]
    targets = [_normalize_header_name(header) for header in target_headers if header]

    for target in targets:
        for doc_header in doc_headers:
            if not target or not doc_header:
                continue
            if target == doc_header or target in doc_header or doc_header in target:
                return True
    return False


def _filter_docs_by_headers(docs: List[Document], target_headers: List[str]) -> List[Document]:
    return [doc for doc in docs if _doc_matches_headers(doc, target_headers)]


def _header_priority(doc: Document) -> int:
    metadata = doc.metadata or {}
    header = _normalize_header_name(metadata.get("section_header", ""))
    header2 = _normalize_header_name(metadata.get("Header 2", ""))

    if header == "project overview" or header2 == "project overview":
        return 0
    if header == "key highlights" or header2 == "key highlights":
        return 1
    if header2 == "more project":
        return 2
    if metadata.get("page_title") and _normalize_header_name(metadata.get("page_title")) == header:
        return 2
    return 3


def _expand_scoped_docs_to_page_context(scoped_docs: List[Document], all_docs: List[Document]) -> List[Document]:
    """
    Once a title header identifies the right page, expand to sibling sections on
    that page so the LLM sees the descriptive chunks, not just the title row.
    """
    if not scoped_docs:
        return []

    sources = []
    seen_sources = set()
    for doc in scoped_docs:
        source = (doc.metadata or {}).get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        sources.append(source)

    expanded: List[Document] = []
    seen_keys = set()
    for source in sources:
        page_docs = [doc for doc in all_docs if (doc.metadata or {}).get("source") == source]
        page_docs = sorted(page_docs, key=_header_priority)
        for doc in page_docs:
            key = _doc_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expanded.append(doc)
    return expanded or scoped_docs


def _annotate_docs(
    docs: List[Document],
    query: str,
    score: float,
    matched_labels: List[str],
) -> List[Tuple[Document, float, List[str]]]:
    annotated: List[Tuple[Document, float, List[str]]] = []
    for doc in docs:
        annotated_doc = copy.deepcopy(doc)
        metadata = annotated_doc.metadata or {}
        if "retrieval_query" not in metadata:
            metadata["retrieval_query"] = query
        if "matched_labels" not in metadata and matched_labels:
            metadata["matched_labels"] = matched_labels
        annotated_doc.metadata = metadata
        annotated.append((annotated_doc, score, matched_labels))
    return annotated


def _rank_docs_for_query(
    query: str,
    candidate_labels: List[str],
    docs: List[Document],
    target_headers: List[str],
    vector_retriever: Any,
    top_k_labels: int,
    label_score_threshold: float,
    vector_fallback_k: int,
) -> List[Tuple[Document, float, List[str]]]:
    scoped_docs = _filter_docs_by_headers(docs, target_headers)
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        logger.info(
            "retrieval.subquery_skipped | %s",
            {
                "subquery": query,
                "target_headers": target_headers,
                "reason": "empty_query",
            },
        )
        return []

    # If the planner already scoped to explicit headers, trust that scope first and
    # rank directly within those docs instead of asking BART to re-select labels.
    if target_headers and scoped_docs:
        expanded_docs = _expand_scoped_docs_to_page_context(scoped_docs, docs)
        logger.info(
            "retrieval.scoped_header_hits | %s",
            {
                "subquery": normalized_query,
                "target_headers": target_headers,
                "match_count": len(scoped_docs),
                "expanded_count": len(expanded_docs),
                "top_sources": [
                    (doc.metadata or {}).get("source", "unknown")
                    for doc in expanded_docs[:3]
                ],
            },
        )
        return _annotate_docs(
            expanded_docs,
            query=normalized_query,
            score=1.0,
            matched_labels=target_headers,
        )

    scoped_labels = build_label_vocab(scoped_docs) if scoped_docs else []
    mapped = map_query_to_labels_zero_shot(
        normalized_query,
        scoped_labels or candidate_labels,
        top_k=top_k_labels,
        score_threshold=label_score_threshold,
    )
    logger.info(
        "retrieval.subquery_labels | %s",
        {
            "subquery": query,
            "target_headers": target_headers,
            "selected_labels": [
                {"label": label, "score": round(score, 4)}
                for label, score in mapped
            ],
        },
    )
    ranked = fetch_docs_by_labels_with_scores(mapped, scoped_docs or docs)
    if ranked:
        logger.info(
            "retrieval.subquery_hits | %s",
            {
                "subquery": query,
                "match_count": len(ranked),
                "top_sources": [
                    (doc.metadata or {}).get("source", "unknown")
                    for doc, _, _ in ranked[:3]
                ],
            },
        )
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    if vector_retriever is None:
        logger.info(
            "retrieval.subquery_no_hits | %s",
            {"subquery": query, "fallback": "none"},
        )
        return []

    try:
        vector_docs = vector_retriever.get_relevant_documents(normalized_query)
    except Exception:
        logger.exception(
            "retrieval.subquery_vector_error | %s",
            {"subquery": query},
        )
        return []

    filtered_vector_docs = _filter_docs_by_headers(vector_docs, target_headers)
    fallback_docs = filtered_vector_docs or vector_docs

    logger.info(
        "retrieval.subquery_vector_fallback | %s",
        {
            "subquery": query,
            "target_headers": target_headers,
            "fallback_count": min(len(fallback_docs), vector_fallback_k),
            "top_sources": [
                (doc.metadata or {}).get("source", "unknown")
                for doc in fallback_docs[:vector_fallback_k]
            ],
        },
    )
    return _annotate_docs(
        fallback_docs[:vector_fallback_k],
        query=query,
        score=0.0,
        matched_labels=[],
    )

class LabelRoutingRetriever(BaseRetriever):
    """
    Retriever that:
      1) Uses BART-MNLI to map query -> section labels.
      2) Fetches all docs whose header/label match those labels.
      3) Ranks docs by label confidence.
      4) Falls back to vector retriever if no labels match.
    """

    docs: List[Document]
    vector_retriever: Any = None
    top_k_labels: int = 5
    label_score_threshold: float = 0.35
    k_docs: int = 6
    vector_fallback_k: int = 2

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        docs: List[Document],
        vector_retriever: Any = None,
        top_k_labels: int = 5,
        label_score_threshold: float = 0.35,
        k_docs: int = 6,
        vector_fallback_k: int = 2,
        **kwargs,
    ):
        super().__init__(
            docs=docs,
            vector_retriever=vector_retriever,
            top_k_labels=top_k_labels,
            label_score_threshold=label_score_threshold,
            k_docs=k_docs,
            vector_fallback_k=vector_fallback_k,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        candidate_labels = build_label_vocab(self.docs)
        header_catalog = build_header_catalog(self.docs)
        plan = plan_retrieval(query, header_catalog)
        logger.info(
            "retrieval.plan | %s",
            {
                "query": query,
                "strategy": plan.strategy,
                "reasoning": plan.reasoning,
                "steps": [step.dict() for step in plan.steps],
                "candidate_label_count": len(candidate_labels),
            },
        )

        step_outputs: Dict[str, Any] = {}
        extracted_values: Dict[str, str] = {}
        docs_with_scores: List[Tuple[Document, float, List[str]]] = []
        ranked_groups: List[List[Tuple[Document, float, List[str]]]] = []

        for step in plan.steps:
            if step.action == "extract":
                source_docs = step_outputs.get(step.depends_on, [])
                entities = extract_entities_from_docs(source_docs, step.extract_fields)
                extracted_map = entities_to_map(entities)
                extracted_values.update({key: value for key, value in extracted_map.items() if value})
                step_outputs[step.step_id] = entities
                logger.info(
                    "retrieval.extract_step | %s",
                    {
                        "step_id": step.step_id,
                        "depends_on": step.depends_on,
                        "fields": step.extract_fields,
                        "entities": entities.dict(),
                    },
                )
                continue

            rendered_query = step.query.format_map(_SafeFormatDict(extracted_values))
            ranked = _rank_docs_for_query(
                rendered_query,
                candidate_labels,
                self.docs,
                step.target_headers,
                self.vector_retriever,
                self.top_k_labels,
                self.label_score_threshold,
                self.vector_fallback_k,
            )
            step_outputs[step.step_id] = [doc for doc, _, _ in ranked]
            if ranked:
                ranked_groups.append(ranked)
                docs_with_scores.extend(ranked)
            logger.info(
                "retrieval.retrieve_step | %s",
                {
                    "step_id": step.step_id,
                    "target_headers": step.target_headers,
                    "query": rendered_query,
                    "result_count": len(ranked),
                },
            )

        if not docs_with_scores and self.vector_retriever is not None:
            vec_docs = self.vector_retriever.get_relevant_documents(query)
            logger.info(
                "retrieval.query_level_vector_fallback | %s",
                {
                    "query": query,
                    "fallback_count": min(len(vec_docs), self.k_docs),
                },
            )
            return vec_docs[: self.k_docs]

        seen_keys = set()
        ranked_docs: List[Document] = []

        # First pass: take the top surviving document from each subquery group so
        # multi-part questions can cover multiple headers before score-only fill.
        for group in ranked_groups:
            for d, sc, matched in group:
                if d.metadata.get("Header 2") == 'More Project':
                    continue
                key = _doc_key(d)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                ranked_docs.append(d)
                break
            if len(ranked_docs) >= self.k_docs:
                return ranked_docs[: self.k_docs]

        # Second pass: fill remaining slots by overall score, still deduped.
        for d, sc, matched in sorted(docs_with_scores, key=lambda x: x[1], reverse=True):
            if d.metadata.get("Header 2") == 'More Project':
                continue
            key = _doc_key(d)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ranked_docs.append(d)
            if len(ranked_docs) >= self.k_docs:
                break
        logger.info(
            "retrieval.final_docs | %s",
            {
                "query": query,
                "returned_count": len(ranked_docs),
                "plan_strategy": plan.strategy,
                "sources": [
                    (doc.metadata or {}).get("source", "unknown")
                    for doc in ranked_docs
                ],
                "headers": [
                    (doc.metadata or {}).get("section_header")
                    or (doc.metadata or {}).get("section_label")
                    or "unknown"
                    for doc in ranked_docs
                ],
            },
        )
        return ranked_docs


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # simple async wrapper
        return self.get_relevant_documents(query)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from .models_groq import get_answer_llm

SYSTEM_PROMPT = """You are Ritam's personal QA bot.
                Use the following context from his website and resume to answer.

                Question: {input}
                Context:
                {context}

                Answer in first person as Ritam."""

def build_rag_chain(vs, k=5, max_docs=6):
    
    all_docs = list(vs.docstore._dict.values())
    vector_retriever = vs.as_retriever(search_kwargs={"k": 8})

    # old: ensemble/PrefixRetriever
    # new: label routing retriever
    retriever = LabelRoutingRetriever(
        docs=all_docs,
        vector_retriever=vector_retriever,
        top_k_labels=k,
        label_score_threshold=0.4,
        k_docs=max_docs,
        vector_fallback_k=2,
    )

    prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPT
            )
    
    llm = get_answer_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain, retriever, SYSTEM_PROMPT

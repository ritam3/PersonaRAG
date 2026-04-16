import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import Field
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import (
    CustomQueryEngine,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.tools import QueryEngineTool
from rank_bm25 import BM25Okapi

from rag_core.embeddings_model import get_embeddings
from rag_core.models_groq import get_answer_llm


logger = logging.getLogger("model_flow")


@dataclass
class ContextDocument:
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class ToolRoutingProfile:
    name: str
    description: str
    profile_text: str
    tokens: List[str]


@dataclass
class RoutingClarification:
    answer: str
    candidate_tools: List[str]
    preferred_tool: str
    original_question: str


def _doc_key(doc: ContextDocument) -> Tuple[str, str]:
    return ((doc.metadata or {}).get("source", ""), doc.page_content[:200])


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _query_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [token for token in tokens if len(token) > 2]


def _metadata_search_text(doc: ContextDocument) -> str:
    metadata = doc.metadata or {}
    return " ".join(
        str(value)
        for value in [
            metadata.get("page_title", ""),
            metadata.get("section_header", ""),
            metadata.get("section_label", ""),
            metadata.get("project_name", ""),
            metadata.get("page_description", ""),
            metadata.get("project_description", ""),
            metadata.get("section_description", ""),
            metadata.get("chunk_description", ""),
        ]
        if value
    )


def _lexical_score(query: str, doc: ContextDocument) -> float:
    query_norm = _normalize(query)
    tokens = _query_tokens(query_norm)
    metadata_text = _metadata_search_text(doc)
    search_text = _normalize(" ".join([metadata_text, doc.page_content[:500]]))
    title = (doc.metadata or {}).get("page_title", "")
    header = (doc.metadata or {}).get("section_header", "")
    label = (doc.metadata or {}).get("section_label", "")
    project_name = (doc.metadata or {}).get("project_name", "")

    score = 0.0
    if query_norm and query_norm in search_text:
        score += 5.0

    for token in tokens:
        if token in _normalize(title):
            score += 2.0
        elif token in _normalize(header):
            score += 1.8
        elif token in _normalize(label):
            score += 1.8
        elif token in _normalize(project_name):
            score += 2.2
        elif token in search_text:
            score += 0.35

    return score


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    if hasattr(response, "response"):
        return str(response.response).strip()
    return str(response).strip()


def _vector_to_doc(node_with_score: Any) -> ContextDocument:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = dict(getattr(node, "metadata", {}) or {})
    text = getattr(node, "text", "") or getattr(node, "get_content", lambda: "")()
    return ContextDocument(page_content=text, metadata=metadata)


def _doc_to_node_with_score(doc: ContextDocument, score: float) -> NodeWithScore:
    node = TextNode(text=doc.page_content, metadata=dict(doc.metadata or {}))
    return NodeWithScore(node=node, score=score)


def _node_with_score_to_doc(node_with_score: NodeWithScore) -> ContextDocument:
    node = node_with_score.node
    return ContextDocument(
        page_content=getattr(node, "text", "") or node.get_content(),
        metadata=dict(getattr(node, "metadata", {}) or {}),
    )


def _doc_matches_specific_project(query: str, doc: ContextDocument) -> bool:
    query_norm = _normalize(query)
    metadata = doc.metadata or {}
    candidates = [
        metadata.get("page_title", ""),
        metadata.get("project_name", ""),
    ]
    return any(candidate and _normalize(candidate) in query_norm for candidate in candidates)


class FocusedSectionQueryEngine(CustomQueryEngine):
    name: str
    description: str
    docs: List[ContextDocument] = Field(default_factory=list)
    llm: Any = Field(exclude=True)
    vector_retriever: Any = Field(default=None, exclude=True)
    top_k: int = 4
    bm25: Any = Field(default=None, exclude=True)
    bm25_docs: List[ContextDocument] = Field(default_factory=list, exclude=True)

    def _allowed_keys(self) -> set[Tuple[str, str]]:
        return {_doc_key(doc) for doc in self.docs}

    def _vector_candidates(self, query: str) -> List[Tuple[ContextDocument, float]]:
        if self.vector_retriever is None:
            return []

        allowed_keys = self._allowed_keys()
        candidates: List[Tuple[ContextDocument, float]] = []
        seen = set()
        for result in self.vector_retriever.retrieve(query):
            doc = _vector_to_doc(result)
            key = _doc_key(doc)
            if key not in allowed_keys or key in seen:
                continue
            seen.add(key)
            candidates.append((doc, float(getattr(result, "score", 0.0) or 0.0)))
        return candidates

    def _bm25_candidates(self, query: str) -> List[Tuple[ContextDocument, float]]:
        if self.bm25 is None or not self.bm25_docs:
            return []

        query_tokens = _query_tokens(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.bm25_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            (doc, float(score))
            for doc, score in ranked[: max(self.top_k * 2, self.top_k)]
            if float(score) > 0
        ]

    def _rank_docs(self, query: str) -> List[Tuple[ContextDocument, float]]:
        combined_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for doc, score in self._bm25_candidates(query):
            key = _doc_key(doc)
            entry = combined_scores.setdefault(key, {"doc": doc, "bm25": 0.0, "vector": 0.0, "lexical": 0.0})
            entry["bm25"] = max(entry["bm25"], float(score))

        for doc, score in self._vector_candidates(query):
            key = _doc_key(doc)
            entry = combined_scores.setdefault(key, {"doc": doc, "bm25": 0.0, "vector": 0.0, "lexical": 0.0})
            entry["vector"] = max(entry["vector"], float(score))

        for doc in self.docs:
            score = _lexical_score(query, doc)
            if score <= 0:
                continue
            key = _doc_key(doc)
            entry = combined_scores.setdefault(key, {"doc": doc, "bm25": 0.0, "vector": 0.0, "lexical": 0.0})
            entry["lexical"] = max(entry["lexical"], float(score))

        if not combined_scores:
            return [(doc, 0.0) for doc in self.docs[: self.top_k]]

        max_bm25 = max((entry["bm25"] for entry in combined_scores.values()), default=0.0) or 1.0
        max_vector = max((entry["vector"] for entry in combined_scores.values()), default=0.0) or 1.0
        max_lexical = max((entry["lexical"] for entry in combined_scores.values()), default=0.0) or 1.0

        ranked: List[Tuple[ContextDocument, float]] = []
        for entry in combined_scores.values():
            normalized_bm25 = entry["bm25"] / max_bm25 if entry["bm25"] else 0.0
            normalized_vector = entry["vector"] / max_vector if entry["vector"] else 0.0
            normalized_lexical = entry["lexical"] / max_lexical if entry["lexical"] else 0.0
            final_score = (
                0.50 * normalized_bm25
                + 0.35 * normalized_vector
                + 0.15 * normalized_lexical
            )
            ranked.append((entry["doc"], final_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

    def custom_query(self, query_str: str) -> Response:
        ranked_docs = self._rank_docs(query_str)
        context_docs = [doc for doc, _score in ranked_docs]
        source_nodes = [_doc_to_node_with_score(doc, score) for doc, score in ranked_docs]

        if not context_docs:
            return Response(
                response="I don't know based on my current retrieved context.",
                source_nodes=[],
                metadata={"tool": self.name},
            )

        context_block = "\n\n".join(
            f"[DOC {idx}] source={(doc.metadata or {}).get('source', 'unknown')} | "
            f"header={(doc.metadata or {}).get('section_header', 'unknown')}\n"
            f"{doc.page_content}"
            for idx, doc in enumerate(context_docs, start=1)
        )

        prompt = (
            "You are answering a focused question about Ritam's background.\n"
            f"Focus area: {self.description}\n\n"
            "Rules:\n"
            "- Answer in first person as Ritam.\n"
            "- Use ONLY the provided context.\n"
            '- If the context is insufficient, say "I don\'t know".\n'
            "- Keep the answer concise and specific.\n\n"
            f"Question:\n{query_str}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )

        answer = _response_text(self.llm.complete(prompt))
        return Response(
            response=answer,
            source_nodes=source_nodes,
            metadata={"tool": self.name},
        )


SYSTEM_PROMPT = """
You are Ritam's personal QA bot.

You will receive:
- `context`: retrieved sections from Ritam's portfolio, project pages, and career materials.
- `chat_history`: prior conversation turns for follow-up understanding only.

Rules:
- Answer in first person as Ritam.
- Use ONLY the provided context for factual claims.
- If the answer is not clearly supported by the context, say "I don't know".
- If the user asks something unrelated to Ritam's career, projects, research, or education,
  politely refuse and ask them to keep the conversation career-related.
- Prefer concise, evidence-backed answers.
""".strip()


class LlamaIndexToolRAGChainCompat:
    ROUTING_CONFIDENCE_THRESHOLD = 0.50
    CLARIFICATION_SCORE_GAP = 0.10

    def __init__(
        self,
        tools: Sequence[QueryEngineTool],
        tool_profiles: Sequence[ToolRoutingProfile],
        system_prompt: str,
        project_titles: Sequence[str] | None = None,
    ) -> None:
        self.tools = list(tools)
        self.tool_map = {tool.metadata.name: tool.query_engine for tool in self.tools}
        self.tool_profiles = list(tool_profiles)
        self.system_prompt = system_prompt
        self.project_titles = [
            _normalize(title)
            for title in (project_titles or [])
            if _normalize(title)
        ]

        bm25_corpus = [profile.tokens for profile in self.tool_profiles if profile.tokens]
        self.routing_profiles_for_bm25 = [
            profile for profile in self.tool_profiles if profile.tokens
        ]
        self.routing_bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None
        self.tool_profile_map = {profile.name: profile for profile in self.tool_profiles}
        self.routing_embed_model = None
        self.tool_profile_embeddings: Dict[str, List[float]] = {}

        try:
            self.routing_embed_model = get_embeddings()
            for profile in self.tool_profiles:
                if not profile.profile_text.strip():
                    continue
                self.tool_profile_embeddings[profile.name] = self.routing_embed_model.get_text_embedding(
                    profile.profile_text
                )
        except Exception as exc:
            logger.warning(
                "retrieval.router_embeddings_unavailable | %s",
                {"error": str(exc)},
            )
            self.routing_embed_model = None
            self.tool_profile_embeddings = {}

    @staticmethod
    def _embedding_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        return float(sum(float(a) * float(b) for a, b in zip(left, right)))

    @staticmethod
    def _is_about_self_query(question: str) -> bool:
        q = _normalize(question)
        direct_phrases = [
            "about yourself",
            "about you",
            "introduce yourself",
            "who are you",
            "your profile",
            "background summary",
            "summary about you",
        ]
        if any(phrase in q for phrase in direct_phrases):
            return True

        return bool(
            re.search(r"\btell me about (you|yourself)\b", q)
            or re.search(r"\bcan you introduce yourself\b", q)
        )

    @staticmethod
    def _contains_any(question: str, phrases: Sequence[str]) -> bool:
        q = _normalize(question)
        return any(phrase in q for phrase in phrases)

    def _has_project_title_match(self, question: str) -> bool:
        q = _normalize(question)
        return any(title and title in q for title in self.project_titles)

    def _route_overrides(self, question: str) -> Dict[str, float]:
        q = _normalize(question)
        boosts: Dict[str, float] = {}

        def boost(tool_name: str, value: float) -> None:
            boosts[tool_name] = max(boosts.get(tool_name, 0.0), value)

        if self._is_about_self_query(q):
            boost("about", 1.0)

        if self._has_project_title_match(q):
            boost("project_detail", 1.0)

        if self._contains_any(
            q,
            [
                "project",
                "projects",
                "portfolio",
                "built",
                "build",
                "worked on",
                "latest works",
            ],
        ):
            boost("projects", 0.85)

        if self._contains_any(
            q,
            [
                "company",
                "companies",
                "employer",
                "employers",
                "role",
                "roles",
                "work experience",
                "career history",
                "time at",
                "worked at",
                "juniper",
                "paytm",
                "mist",
                "internship",
                "full time",
            ],
        ):
            boost("experience", 0.9)

        if self._contains_any(
            q,
            [
                "research",
                "paper",
                "papers",
                "publication",
                "publications",
                "lab",
                "acl",
            ],
        ):
            boost("research", 0.9)

        if self._contains_any(
            q,
            [
                "education",
                "degree",
                "degrees",
                "university",
                "universities",
                "college",
                "coursework",
                "academic background",
                "asu",
            ],
        ):
            boost("education", 0.9)

        if not self._is_about_self_query(q):
            boosts["about"] = 0.0

        return boosts

    def _route_tool_scores(self, question: str) -> List[Tuple[str, float]]:
        query_tokens = _query_tokens(question)
        overrides = self._route_overrides(question)
        combined_scores: Dict[str, Dict[str, float]] = {
            profile.name: {"bm25": 0.0, "embedding": 0.0, "lexical": 0.0}
            for profile in self.tool_profiles
        }

        if self.routing_bm25 is not None and query_tokens:
            bm25_scores = self.routing_bm25.get_scores(query_tokens)
            for profile, score in zip(self.routing_profiles_for_bm25, bm25_scores):
                combined_scores[profile.name]["bm25"] = float(score)

        if self.routing_embed_model is not None and self.tool_profile_embeddings:
            try:
                query_embedding = self.routing_embed_model.get_query_embedding(question)
                for profile in self.tool_profiles:
                    profile_embedding = self.tool_profile_embeddings.get(profile.name)
                    if not profile_embedding:
                        continue
                    combined_scores[profile.name]["embedding"] = max(
                        0.0,
                        self._embedding_similarity(query_embedding, profile_embedding),
                    )
            except Exception as exc:
                logger.warning(
                    "retrieval.router_query_embedding_failed | %s",
                    {"error": str(exc)},
                )

        query_norm = _normalize(question)
        for profile in self.tool_profiles:
            profile_text_norm = _normalize(profile.profile_text)
            lexical = 0.0
            if query_norm and query_norm in profile_text_norm:
                lexical += 4.0
            for token in query_tokens:
                if token in profile.tokens:
                    lexical += 1.0
                elif token in profile_text_norm:
                    lexical += 0.25
            combined_scores[profile.name]["lexical"] = lexical

        max_bm25 = max((scores["bm25"] for scores in combined_scores.values()), default=0.0) or 1.0
        max_embedding = max((scores["embedding"] for scores in combined_scores.values()), default=0.0) or 1.0
        max_lexical = max((scores["lexical"] for scores in combined_scores.values()), default=0.0) or 1.0

        raw_ranked: List[Tuple[str, float]] = []
        for profile in self.tool_profiles:
            scores = combined_scores[profile.name]
            semantic_score = (
                0.35 * (scores["bm25"] / max_bm25 if scores["bm25"] else 0.0)
                + 0.50 * (scores["embedding"] / max_embedding if scores["embedding"] else 0.0)
                + 0.15 * (scores["lexical"] / max_lexical if scores["lexical"] else 0.0)
            )
            override_score = overrides.get(profile.name, 0.0)
            if profile.name == "about" and not self._is_about_self_query(question):
                semantic_score = 0.0
            final = max(semantic_score, override_score)
            raw_ranked.append((profile.name, final))

        temperature = 3.0
        exp_scores = {
            tool_name: math.exp(score * temperature)
            for tool_name, score in raw_ranked
        }
        score_total = sum(exp_scores.values()) or 1.0
        ranked = [
            (tool_name, exp_scores[tool_name] / score_total)
            for tool_name, _score in raw_ranked
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _high_confidence_tools(
        self,
        ranked_tools: Sequence[Tuple[str, float]],
        threshold: float | None = None,
    ) -> List[Tuple[str, float]]:
        min_score = self.ROUTING_CONFIDENCE_THRESHOLD if threshold is None else threshold
        selected = [
            (tool_name, score)
            for tool_name, score in ranked_tools
            if score >= min_score
        ]
        return selected or list(ranked_tools[:1])

    def _is_multi_part_query(self, question: str, ranked_tools: Sequence[Tuple[str, float]]) -> bool:
        q = _normalize(question)
        multi_markers = [
            " and ",
            " also ",
            " as well as ",
            " along with ",
            " compare ",
            " compared to ",
            " versus ",
            " vs ",
            " both ",
            " difference ",
        ]
        has_multi_marker = any(marker in q for marker in multi_markers)
        if len(ranked_tools) < 2:
            return False

        top_score = ranked_tools[0][1]
        second_score = ranked_tools[1][1]
        if has_multi_marker and second_score >= max(0.35, top_score * 0.65):
            return True
        return False

    def _tool_display_name(self, tool_name: str) -> str:
        labels = {
            "about": "a broader background summary",
            "experience": "work experience",
            "projects": "projects",
            "education": "education",
            "research": "research",
            "project_detail": "a specific project",
        }
        return labels.get(tool_name, tool_name.replace("_", " "))

    def _clarification_prompt(
        self,
        broad_tool: str,
        specific_tool: str,
        ranked_tools: Sequence[Tuple[str, float]],
    ) -> str:
        specific_label = self._tool_display_name(specific_tool)
        broad_label = self._tool_display_name(broad_tool)
        del ranked_tools
        return (
            f"Do you want {specific_label} specifically, or {broad_label}? "
            f"Reply with something short like '{specific_tool.replace('_', ' ')}' or 'broader summary'."
        )

    def _maybe_build_clarification(
        self,
        question: str,
        ranked_tools: Sequence[Tuple[str, float]],
    ) -> Optional[RoutingClarification]:
        if len(ranked_tools) < 2:
            return None

        top_tool, top_score = ranked_tools[0]
        second_tool, second_score = ranked_tools[1]
        if top_tool != "about" or second_tool == "about":
            return None

        score_gap = top_score - second_score
        if score_gap > self.CLARIFICATION_SCORE_GAP and second_score < top_score * 0.82:
            return None

        return RoutingClarification(
            answer=self._clarification_prompt(
                broad_tool=top_tool,
                specific_tool=second_tool,
                ranked_tools=ranked_tools,
            ),
            candidate_tools=[second_tool, top_tool],
            preferred_tool=second_tool,
            original_question=question,
        )

    def _select_tool_name(self, ranked_tools: Sequence[Tuple[str, float]]) -> str:
        for tool_name, _score in ranked_tools:
            if tool_name in self.tool_map:
                return tool_name
        return "about"

    def _select_tool_names_for_multi(
        self,
        ranked_tools: Sequence[Tuple[str, float]],
    ) -> List[str]:
        if not ranked_tools:
            return ["about"]

        top_score = ranked_tools[0][1]
        selected = [
            tool_name
            for tool_name, score in ranked_tools[:3]
            if tool_name in self.tool_map and score >= max(0.35, top_score * 0.65)
        ]
        return selected or [self._select_tool_name(ranked_tools)]

    def _run_tool(self, tool_name: str, question: str) -> Response:
        query_engine = self.tool_map[tool_name]
        return query_engine.query(question)

    def resolve_clarification_reply(
        self,
        reply: str,
        candidate_tools: Sequence[str],
        preferred_tool: str,
    ) -> str:
        reply_norm = _normalize(reply)
        specific_markers = {
            "yes",
            "yep",
            "yeah",
            "specific",
            "specifically",
            "detailed",
            "detail",
            "that one",
        }
        broad_markers = {
            "broad",
            "broader",
            "summary",
            "overview",
            "general",
            "background",
        }

        if reply_norm in specific_markers or any(marker in reply_norm for marker in specific_markers):
            return preferred_tool
        if any(marker in reply_norm for marker in broad_markers):
            for tool_name in candidate_tools:
                if tool_name == "about":
                    return tool_name

        filtered_ranked = [
            (tool_name, score)
            for tool_name, score in self._route_tool_scores(reply)
            if tool_name in candidate_tools
        ]
        if filtered_ranked:
            return filtered_ranked[0][0]
        return preferred_tool

    def _combine_tool_responses(
        self,
        responses: Sequence[Response],
        question: str,
    ) -> Response:
        source_nodes: List[NodeWithScore] = []
        seen = set()
        partials: List[str] = []
        for response in responses:
            answer_text = _response_text(response)
            if answer_text:
                partials.append(answer_text)
            for node in getattr(response, "source_nodes", []) or []:
                doc = _node_with_score_to_doc(node)
                key = _doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                source_nodes.append(node)

        if not source_nodes:
            return Response(
                response="I don't know based on my current retrieved context.",
                source_nodes=[],
                metadata={"engine": "manual_multi"},
            )

        context_docs = [_node_with_score_to_doc(node) for node in source_nodes]
        context_block = "\n\n".join(
            f"[DOC {idx}] source={(doc.metadata or {}).get('source', 'unknown')} | "
            f"header={(doc.metadata or {}).get('section_header', 'unknown')}\n"
            f"{doc.page_content}"
            for idx, doc in enumerate(context_docs, start=1)
        )
        partial_block = "\n\n".join(
            f"[PARTIAL {idx}] {text}" for idx, text in enumerate(partials, start=1)
        )

        prompt = (
            f"{self.system_prompt}\n\n"
            f"Question:\n{question}\n\n"
            f"Partial answers from specialized tools:\n{partial_block}\n\n"
            f"Context:\n{context_block}\n\n"
            "Synthesize a final answer that directly answers the question."
        )
        final_answer = _response_text(get_answer_llm().complete(prompt))
        return Response(
            response=final_answer,
            source_nodes=source_nodes,
            metadata={"engine": "manual_multi"},
        )

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = payload.get("input", "")
        forced_tool = payload.get("forced_tool")
        ranked_tools = self._route_tool_scores(question)
        if forced_tool in self.tool_map:
            response = self._run_tool(forced_tool, question)
            engine_used = f"forced_router:{forced_tool}"
        else:
            clarification = self._maybe_build_clarification(question, ranked_tools)
            if clarification is not None:
                high_confidence_tools = self._high_confidence_tools(ranked_tools)
                logger.info(
                    "retrieval.router_clarification | %s",
                    {
                        "query": question,
                        "ranked_tools": ranked_tools,
                        "high_confidence_tools": high_confidence_tools,
                        "candidate_tools": clarification.candidate_tools,
                    },
                )
                return {
                    "answer": clarification.answer,
                    "context": [],
                    "needs_clarification": True,
                    "candidate_tools": clarification.candidate_tools,
                    "preferred_tool": clarification.preferred_tool,
                    "original_question": clarification.original_question,
                    "ranked_tools": high_confidence_tools,
                }
            if self._is_multi_part_query(question, ranked_tools):
                selected_tool_names = self._select_tool_names_for_multi(ranked_tools)
                partial_responses = [
                    self._run_tool(tool_name, question)
                    for tool_name in selected_tool_names
                ]
                response = self._combine_tool_responses(partial_responses, question)
                engine_used = "manual_multi"
            else:
                tool_name = self._select_tool_name(ranked_tools)
                response = self._run_tool(tool_name, question)
                engine_used = f"local_router:{tool_name}"

        answer = _response_text(response)
        source_nodes = getattr(response, "source_nodes", []) or []
        context_docs = [_node_with_score_to_doc(node) for node in source_nodes]

        logger.info(
            "retrieval.llamaindex_engine | %s",
            {
                "query": question,
                "engine": engine_used,
                "ranked_tools": ranked_tools,
                "high_confidence_tools": self._high_confidence_tools(ranked_tools),
                "source_count": len(context_docs),
            },
        )
        return {"answer": answer, "context": context_docs}


def _docs_for_tool(docs: Sequence[ContextDocument], tool_name: str) -> List[ContextDocument]:
    selected: List[ContextDocument] = []

    for doc in docs:
        metadata = doc.metadata or {}
        section_type = _normalize(metadata.get("section_type", ""))
        page_type = _normalize(metadata.get("page_type", ""))
        header = _normalize(metadata.get("section_header", ""))

        if tool_name == "experience":
            if section_type == "experience" or header == "work experience":
                selected.append(doc)
        elif tool_name == "projects":
            if page_type in {"project_detail", "projects_index"} or section_type == "projects":
                selected.append(doc)
        elif tool_name == "education":
            if section_type == "education":
                selected.append(doc)
        elif tool_name == "research":
            if section_type == "research" or "research" in header:
                selected.append(doc)
        elif tool_name == "project_detail":
            if page_type == "project_detail":
                selected.append(doc)
        elif tool_name == "about":
            if header == "about" or section_type == "about":
                selected.append(doc)

    deduped: List[ContextDocument] = []
    seen = set()
    for doc in selected:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _build_query_engine_tools(
    docs: Sequence[ContextDocument],
    vector_retriever: Any,
) -> Tuple[List[QueryEngineTool], List[ToolRoutingProfile], List[str]]:
    answer_llm = get_answer_llm()

    tool_specs = [
        (
            "experience",
            "Experience",
            "Use for questions about Ritam's work experience, career history, companies, roles, responsibilities, and general experience. This also covers broad experience questions that may include research experience.",
            "Questions about work experience, career history, employers, internships, full-time roles, responsibilities, time at companies, Juniper Networks, Paytm Money, Mist, CoRAL Lab, and what Ritam did at each company.",
            4,
        ),
        (
            "projects",
            "Projects",
            "Use for broad questions about Ritam's projects, portfolio, multiple projects, or which projects are relevant to a role.",
            "Questions about broad project portfolio, kinds of projects, multiple projects, best projects, relevant projects, what Ritam has built, and portfolio work across several projects.",
            4,
        ),
        (
            "education",
            "Education",
            "Use for questions about Ritam's education, degrees, universities, academic background, or coursework-related background.",
            "Questions about education, degrees, universities, Arizona State University, academic background, studies, and coursework.",
            3,
        ),
        (
            "research",
            "Research",
            "Use for questions specifically about research experience, publications, labs, papers, or research-oriented work.",
            "Questions about research experience, papers, publications, ACL work, lab work, benchmarks, experiments, and academic research.",
            3,
        ),
        (
            "project_detail",
            "Project Detail",
            "Use for deep dives into a specific named project, shorthand references to a project, or follow-up questions about one particular project page.",
            "Questions about one specific named project, a deep dive into a single project, project details, technical decisions, stack, implementation, and follow-ups about one project.",
            3,
        ),
        (
            "about",
            "About",
            "Use for questions about Ritam's personal introduction or short background summary from the About section only.",
            "Questions about who Ritam is, introducing himself, personal introduction, short background summary, profile summary, and self introduction.",
            3,
        ),
    ]

    tools: List[QueryEngineTool] = []
    tool_profiles: List[ToolRoutingProfile] = []
    project_titles: List[str] = []

    for tool_name, display_name, description, routing_text, top_k in tool_specs:
        tool_docs = _docs_for_tool(docs, tool_name)
        if not tool_docs:
            continue

        bm25_corpus = []
        bm25_docs: List[ContextDocument] = []
        for doc in tool_docs:
            search_text = " ".join(
                [
                    _metadata_search_text(doc),
                    doc.page_content,
                ]
            )
            tokens = _query_tokens(search_text)
            if not tokens:
                continue
            bm25_corpus.append(tokens)
            bm25_docs.append(doc)

        bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None

        query_engine = FocusedSectionQueryEngine(
            name=tool_name,
            description=description,
            docs=tool_docs,
            llm=answer_llm,
            vector_retriever=vector_retriever,
            top_k=top_k,
            bm25=bm25,
            bm25_docs=bm25_docs,
        )
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=tool_name,
            description=description,
            return_direct=False,
        )
        tools.append(tool)

        logger.info(
            "retrieval.tool_built | %s",
            {
                "tool": tool_name,
                "display_name": display_name,
                "doc_count": len(tool_docs),
            },
        )

        if tool_name == "project_detail":
            for doc in tool_docs:
                metadata = doc.metadata or {}
                for title in [metadata.get("page_title", ""), metadata.get("project_name", "")]:
                    normalized_title = _normalize(str(title))
                    if normalized_title and normalized_title not in project_titles:
                        project_titles.append(normalized_title)

        profile_text = " ".join(
            part for part in [display_name, description, routing_text] if part
        )
        tool_profiles.append(
            ToolRoutingProfile(
                name=tool_name,
                description=description,
                profile_text=profile_text,
                tokens=_query_tokens(profile_text),
            )
        )

    return tools, tool_profiles, project_titles


def build_rag_chain(index, docs, k: int = 5, max_docs: int = 4):
    vector_retriever = index.as_retriever(similarity_top_k=max(k, 8))
    tools, tool_profiles, project_titles = _build_query_engine_tools(docs, vector_retriever)

    rag_chain = LlamaIndexToolRAGChainCompat(
        tools=tools,
        tool_profiles=tool_profiles,
        system_prompt=SYSTEM_PROMPT,
        project_titles=project_titles,
    )
    return rag_chain, tools, SYSTEM_PROMPT

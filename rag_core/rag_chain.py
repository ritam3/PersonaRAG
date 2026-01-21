# rag_core/rag_chain.py (top of file)

from typing import List, Tuple, Dict, Any
from langchain.schema import Document, BaseRetriever
from transformers import pipeline
from rag_core.index_builder import load_vectorstore, build_and_save_index


# ---- Label routing helpers ----

# Global zero-shot classifier (loaded once)
label_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

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

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        docs: List[Document],
        vector_retriever: Any = None,
        top_k_labels: int = 5,
        label_score_threshold: float = 0.35,
        k_docs: int = 6,
        **kwargs,
    ):
        super().__init__(
            docs=docs,
            vector_retriever=vector_retriever,
            top_k_labels=top_k_labels,
            label_score_threshold=label_score_threshold,
            k_docs=k_docs,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1) build candidate labels from docs
        candidate_labels = build_label_vocab(self.docs)

        # 2) map query -> (label, score)
        mapped = map_query_to_labels_zero_shot(
            query,
            candidate_labels,
            top_k=self.top_k_labels,
            score_threshold=self.label_score_threshold,
        )

        # 3) fetch docs with scores
        docs_with_scores = fetch_docs_by_labels_with_scores(mapped, self.docs)

        if not docs_with_scores and self.vector_retriever is not None:
            # fallback to vector retriever (semantic retrieval)
            vec_docs = self.vector_retriever.get_relevant_documents(query)
            return vec_docs[: self.k_docs]

        # 4) sort by score desc + dedupe
        seen_keys = set()
        ranked_docs: List[Document] = []
        for d, sc, matched in sorted(docs_with_scores, key=lambda x: x[1], reverse=True):
            if d.metadata.get("Header 2")=='More Project':
                continue
            key = (d.metadata.get("source"), d.page_content[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ranked_docs.append(d)
            if len(ranked_docs) >= self.k_docs:
                break
        return ranked_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # simple async wrapper
        return self.get_relevant_documents(query)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from .models_groq import get_answer_llm, get_judge_llm

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

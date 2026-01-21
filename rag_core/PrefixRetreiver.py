from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from langchain.schema import Document
import re

def _norm(text: str) -> str:
    # lower, remove punctuation-ish chars, compact whitespace
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PrefixRetriever:
    def __init__(self, docs: List[Document], k: int = 3, max_lines: int = 8):
        self.docs = docs
        self.k = k
        self.max_lines = max_lines

    def _head(self, content: str) -> str:
        # return up to max_lines lines from the beginning to be used for prefix matching
        lines = [ln for ln in content.splitlines() if ln.strip()]
        return "\n".join(lines[: self.max_lines])

    def _get_relevant_documents(self, query: str) -> List[Document]:
        q = _norm(query)
        tokens = [t for t in q.split() if t]
        out: List[Document] = []

        for d in self.docs:
            # build a matching head: prefer explicit header metadata
            header = d.metadata.get("section_header") or d.metadata.get("section_label") or ""
            head_text = header + "\n" + self._head(d.page_content)
            head = _norm(head_text)

            matched = False
            # 1) exact-substring match (most important)
            if q and q in head:
                matched = True
            else:
                # 2) token-based match: ensure every token exists in head
                if tokens and all(tok in head for tok in tokens):
                    matched = True

            if matched:
                out.append(d)
                if len(out) >= self.k:
                    break

        return out


# --- Helper to recover Documents from a FAISS vectorstore
def faiss_all_docs(faiss_store):
    ids = list(faiss_store.index_to_docstore_id.values())
    return [faiss_store.docstore.search(_id) for _id in ids]


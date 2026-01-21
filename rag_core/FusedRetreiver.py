from typing import Any, List, Set, Tuple
from langchain.schema import BaseRetriever, Document

class FusedRetriever(BaseRetriever):
    """
    Pydantic-compatible fused retriever that wraps a header-aware PrefixRetriever
    and a vector retriever. Declares fields so Pydantic validation succeeds.
    """
    prefix_retriever: Any
    vector_retriever: Any
    k: int = 4
    prefix_first: bool = True

    class Config:
        # allows storing arbitrary Python objects in model fields
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1) prefix candidates
        prefix_docs = []
        if hasattr(self.prefix_retriever, "get_relevant_documents"):
            prefix_docs = self.prefix_retriever.get_relevant_documents(query)
        elif hasattr(self.prefix_retriever, "_get_relevant_documents"):
            prefix_docs = self.prefix_retriever._get_relevant_documents(query)

        # 2) vector candidates
        vector_docs = []
        try:
            # many LangChain retrievers implement get_relevant_documents
            if hasattr(self.vector_retriever, "get_relevant_documents"):
                vector_docs = self.vector_retriever.get_relevant_documents(query)
            elif hasattr(self.vector_retriever, "retrieve"):
                vector_docs = self.vector_retriever.retrieve(query)
            elif hasattr(self.vector_retriever, "get_relevant_documents_async"):
                vector_docs = self.vector_retriever.get_relevant_documents_async(query)
        except Exception:
            vector_docs = []

        # fuse with dedupe
        seen: Set[Tuple[str, str]] = set()
        out: List[Document] = []

        def add_docs(docs: List[Document]) -> bool:
            for d in docs:
                key = (d.metadata.get("source"), d.page_content[:200])
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
                if len(out) >= self.k:
                    return True
            return False

        if self.prefix_first:
            finished = add_docs(prefix_docs)
            if not finished:
                add_docs(vector_docs)
        else:
            finished = add_docs(vector_docs)
            if not finished:
                add_docs(prefix_docs)

        return out[: self.k]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Try to call async variants when available, else fall back to sync
        prefix_docs = []
        if hasattr(self.prefix_retriever, "aget_relevant_documents"):
            prefix_docs = await self.prefix_retriever.aget_relevant_documents(query)
        elif hasattr(self.prefix_retriever, "get_relevant_documents"):
            prefix_docs = self.prefix_retriever.get_relevant_documents(query)
        elif hasattr(self.prefix_retriever, "_get_relevant_documents"):
            prefix_docs = self.prefix_retriever._get_relevant_documents(query)

        vector_docs = []
        if hasattr(self.vector_retriever, "aget_relevant_documents"):
            vector_docs = await self.vector_retriever.aget_relevant_documents(query)
        elif hasattr(self.vector_retriever, "get_relevant_documents"):
            vector_docs = self.vector_retriever.get_relevant_documents(query)
        elif hasattr(self.vector_retriever, "retrieve"):
            vector_docs = self.vector_retriever.retrieve(query)

        # Reuse sync fuse logic by delegating to get_relevant_documents after patching
        # Build a temporary 'self' like structure — easiest is to merge candidate lists here
        seen = set()
        out = []
        def add_docs_sync(docs):
            for d in docs:
                key = (d.metadata.get("source"), d.page_content[:200])
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
                if len(out) >= self.k:
                    return True
            return False

        if self.prefix_first:
            finished = add_docs_sync(prefix_docs)
            if not finished:
                add_docs_sync(vector_docs)
        else:
            finished = add_docs_sync(vector_docs)
            if not finished:
                add_docs_sync(prefix_docs)

        return out[: self.k]

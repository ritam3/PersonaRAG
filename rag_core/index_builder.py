import io
import json
import re
from pathlib import Path
from typing import Dict, List

import faiss
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from pypdf import PdfReader

from .config import FAISS_INDEX_PATH, INDEX_CACHE_PATH, INDEX_ID, NODE_CACHE_PATH, VECTORSTORE_PATH
from .crawler import crawl_subpages
from .embeddings_model import get_embeddings
from .sources import CRAWL_ROOTS, FIXED_URLS


def _is_gdrive_file(url: str) -> bool:
    return "drive.google.com" in url and "/file/d/" in url


def _gdrive_view_to_download(url: str) -> str:
    match = re.search(r"/file/d/([^/]+)/", url)
    if not match:
        return url
    file_id = match.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _extract_summary_sentences(
    text: str,
    max_sentences: int = 2,
    max_chars: int = 320,
) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    picked: List[str] = []
    for sentence in sentences:
        clean = sentence.strip()
        if len(clean) < 30:
            continue
        lowered = clean.lower()
        if lowered in {"project overview", "key highlights", "more project"}:
            continue
        picked.append(clean)
        if len(picked) >= max_sentences:
            break

    if not picked:
        summary = normalized[:max_chars].rstrip()
    else:
        summary = " ".join(picked)

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _is_project_page(url: str) -> bool:
    return "/project/" in (url or "").rstrip("/")


def _is_project_index_page(url: str) -> bool:
    normalized = (url or "").rstrip("/")
    return normalized.endswith("/project")


def _is_noise_header(header: str) -> bool:
    return _normalize_whitespace(header).lower() in {
        "",
        "start of bodystart",
        "project overview",
        "key highlights",
        "more project",
    }


def _is_site_chrome_header(header: str) -> bool:
    normalized = _normalize_whitespace(header).lower()
    return normalized in {
        "",
        "ritam's portfolio",
        "back to homepage",
        "search menu",
        "available for work",
    }


def _clean_title_candidate(header: str) -> str:
    cleaned = _normalize_whitespace(header)
    cleaned = re.sub(r"\s*[-|:]+\s*$", "", cleaned).strip()
    return cleaned


def _is_special_project_h2(header: str) -> bool:
    normalized = _normalize_whitespace(header).lower()
    return normalized in {
        "project overview",
        "key highlights",
        "more project",
    }


def _derive_page_title(records_for_page: List[dict]) -> str:
    for record in records_for_page:
        metadata = record["metadata"]
        header = _normalize_whitespace(metadata.get("section_header", ""))
        header2 = _normalize_whitespace(metadata.get("Header 2", ""))
        header1 = _clean_title_candidate(metadata.get("Header 1", ""))
        if header2.lower() == "more project" and header1 and not _is_site_chrome_header(header1):
            return header1

    for record in records_for_page:
        metadata = record["metadata"]
        header1 = _clean_title_candidate(metadata.get("Header 1", ""))
        if header1 and not _is_site_chrome_header(header1) and not _is_noise_header(header1):
            return header1

    for record in records_for_page:
        metadata = record["metadata"]
        header = _clean_title_candidate(metadata.get("section_header", ""))
        if _is_noise_header(header):
            continue
        if _is_site_chrome_header(header):
            continue
        if _is_project_page(metadata.get("source", "")) and header.lower() in {
            "project overview",
            "key highlights",
        }:
            continue
        return header

    return ""


def _classify_page_type(url: str) -> str:
    normalized = (url or "").lower()
    if "/project/" in normalized:
        return "project_detail"
    if normalized.rstrip("/").endswith("/project"):
        return "projects_index"
    if "old-home" in normalized:
        return "about"
    if "scholar.google.com" in normalized:
        return "research_profile"
    if normalized.rstrip("/").endswith("/stack"):
        return "stack"
    if normalized.rstrip("/").endswith("framer.app") or normalized.rstrip("/").endswith("framer.app/"):
        return "landing"
    return "reference"


def _classify_section_type(header: str, page_type: str) -> str:
    normalized = _normalize_whitespace(header).lower()
    if "research" in normalized:
        return "research"
    if "experience" in normalized:
        return "experience"
    if "education" in normalized:
        return "education"
    if "skill" in normalized or "stack" in normalized:
        return "skills"
    if "project" in normalized and page_type != "project_detail":
        return "projects"
    if page_type == "project_detail":
        if "highlight" in normalized:
            return "project_highlights"
        if "overview" in normalized:
            return "project_overview"
        return "project_detail"
    return "general"


def _project_name_from_url(url: str) -> str:
    if not _is_project_page(url):
        return ""
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def _build_page_summaries(records: List[dict]) -> None:
    records_by_source: Dict[str, List[dict]] = {}
    for record in records:
        source = (record["metadata"] or {}).get("source", "")
        records_by_source.setdefault(source, []).append(record)

    for source, records_for_page in records_by_source.items():
        page_title = _derive_page_title(records_for_page)
        page_type = _classify_page_type(source)
        page_text = "\n\n".join(
            _normalize_whitespace(record.get("text", "")) for record in records_for_page
        )
        page_description = _extract_summary_sentences(page_text, max_sentences=3, max_chars=420)

        project_name = _project_name_from_url(source)
        for record in records_for_page:
            metadata = record["metadata"]
            fallback_title = _clean_title_candidate(
                page_title or project_name or metadata.get("section_header", "")
            )
            section_header = _clean_title_candidate(metadata.get("section_header", ""))
            metadata["page_title"] = fallback_title
            metadata["page_type"] = page_type
            metadata["project_name"] = project_name
            metadata["section_type"] = _classify_section_type(
                metadata.get("section_header", ""),
                page_type,
            )
            if section_header and section_header != fallback_title:
                metadata["section_label"] = f"{fallback_title} :: {section_header}"
            else:
                metadata["section_label"] = section_header or fallback_title
            metadata["page_description"] = page_description
            metadata["section_description"] = _extract_summary_sentences(
                record.get("text", ""),
                max_sentences=2,
                max_chars=320,
            )
            metadata["chunk_description"] = metadata["section_description"]
            if page_type == "project_detail":
                metadata["project_description"] = page_description


def _section_record(
    text: str,
    url: str,
    header_1: str = "",
    header_2: str = "",
) -> dict | None:
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return None

    section_header = header_2 or header_1 or "Page Content"
    metadata = {
        "source": url,
        "section_header": _normalize_whitespace(section_header),
        "section_label": _normalize_whitespace(section_header),
        "section_type": "general",
        "section_scope": _normalize_whitespace(section_header).lower().replace(" ", "_"),
        "Header 1": _normalize_whitespace(header_1),
        "Header 2": _normalize_whitespace(header_2),
    }
    return {"text": normalized_text, "metadata": metadata}


def _merge_record_text(existing_text: str, incoming_text: str) -> str:
    existing = _normalize_whitespace(existing_text)
    incoming = _normalize_whitespace(incoming_text)
    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming in existing:
        return existing
    return f"{existing}\n\n{incoming}"


def _normalize_records(records: List[dict]) -> List[dict]:
    normalized_records: List[dict] = []
    records_by_source: Dict[str, List[dict]] = {}
    for record in records:
        source = (record.get("metadata") or {}).get("source", "")
        records_by_source.setdefault(source, []).append(record)

    for source, source_records in records_by_source.items():
        page_type = _classify_page_type(source)
        merged_for_source: List[dict] = []
        index_by_key: Dict[tuple, int] = {}

        for record in source_records:
            metadata = dict(record.get("metadata") or {})
            text = _normalize_whitespace(record.get("text", ""))
            header1 = _clean_title_candidate(metadata.get("Header 1", ""))
            header2 = _clean_title_candidate(metadata.get("Header 2", ""))
            section_header = _clean_title_candidate(metadata.get("section_header", ""))

            if _is_site_chrome_header(section_header):
                continue
            if section_header.lower() == "more project":
                continue

            effective_header = section_header or header1 or "Page Content"
            effective_h2 = header2

            if page_type == "project_detail":
                if _is_special_project_h2(header2) or _is_special_project_h2(section_header):
                    effective_header = header1 or section_header or effective_header
                    effective_h2 = ""
                elif header1 and section_header == header1:
                    effective_header = header1
                    effective_h2 = ""

            metadata["section_header"] = effective_header
            metadata["section_label"] = effective_header
            metadata["Header 1"] = header1
            metadata["Header 2"] = effective_h2

            key = (effective_header, effective_h2)
            existing_index = index_by_key.get(key)
            if existing_index is None:
                merged_record = {
                    "text": text,
                    "metadata": metadata,
                }
                index_by_key[key] = len(merged_for_source)
                merged_for_source.append(merged_record)
            else:
                merged_for_source[existing_index]["text"] = _merge_record_text(
                    merged_for_source[existing_index]["text"],
                    text,
                )

        normalized_records.extend(merged_for_source)

    return normalized_records


def _parse_html_sections(html: str, url: str) -> List[dict]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    elements = body.find_all(["h1", "h2", "p", "li"], recursive=True)

    sections: List[dict] = []
    current_h1 = ""
    current_h2 = ""
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        record = _section_record(
            "\n".join(current_lines),
            url=url,
            header_1=current_h1,
            header_2=current_h2,
        )
        if record:
            record["metadata"]["section_type"] = "remote_html"
            sections.append(record)
        current_lines = []

    for element in elements:
        text = _normalize_whitespace(element.get_text(" ", strip=True))
        if not text:
            continue

        if element.name == "h1":
            if current_lines:
                flush_section()
            current_h1 = text
            current_h2 = ""
            current_lines = [text]
            continue

        if element.name == "h2":
            if _is_special_project_h2(text):
                if not current_h1:
                    current_h1 = text
                    current_lines = [current_h1]
                continue
            if current_lines:
                flush_section()
            if not current_h1:
                current_h1 = text
            current_h2 = text
            current_lines = [text]
            continue

        if not current_h1 and element.name in {"p", "li"}:
            title = _normalize_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
            current_h1 = title or _project_name_from_url(url) or "Page Content"
            current_h2 = ""
            current_lines = [current_h1]

        if current_lines:
            current_lines.append(text)

    if current_lines:
        flush_section()

    return sections


def _load_pdf_records(pdf_url: str) -> List[dict]:
    response = requests.get(pdf_url, timeout=20)
    response.raise_for_status()
    reader = PdfReader(io.BytesIO(response.content))

    records: List[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize_whitespace(page.extract_text() or "")
        if not text:
            continue
        records.append(
            {
                "text": text,
                "metadata": {
                    "source": pdf_url,
                    "section_header": f"PDF Page {page_number}",
                    "section_label": f"pdf_page_{page_number}",
                    "section_type": "remote_pdf",
                    "section_scope": "remote_pdf",
                    "Header 1": "",
                    "Header 2": "",
                },
            }
        )
    return records


def load_web_docs(urls: List[str]) -> List[dict]:
    records: List[dict] = []

    for url in urls:
        clean_url = url.strip()
        if not clean_url:
            continue

        target_url = _gdrive_view_to_download(clean_url) if _is_gdrive_file(clean_url) else clean_url
        try:
            if target_url.lower().endswith(".pdf") or _is_gdrive_file(clean_url):
                print(f"[index_builder] Loading PDF from {target_url}")
                records.extend(_load_pdf_records(target_url))
                continue

            print(f"[index_builder] Fetching HTML from {target_url}")
            response = requests.get(target_url, timeout=15)
            response.raise_for_status()
            page_records = _parse_html_sections(response.text, target_url)
            print(f"[index_builder]   {target_url}: {len(page_records)} sections")
            records.extend(page_records)
        except Exception as exc:
            print(f"[index_builder] Error processing {target_url}: {exc}")

    records = _normalize_records(records)
    _build_page_summaries(records)
    return records


def split_docs(records: List[dict]) -> List[dict]:
    """HTML is already section-based; keep each extracted section as one node."""
    print(f"[index_builder] split_docs: {len(records)} section nodes")
    return records


def _persist_node_cache(records: List[dict]) -> None:
    INDEX_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    with NODE_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def load_node_cache() -> List[dict]:
    if not NODE_CACHE_PATH.exists():
        return []
    with NODE_CACHE_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    records = _normalize_records(records)
    _build_page_summaries(records)
    return records


def build_and_save_index():
    """Crawl URLs, extract section nodes, build a LlamaIndex FAISS store, and persist it."""
    crawl_urls: List[str] = []
    for root in CRAWL_ROOTS:
        try:
            urls = crawl_subpages(root)
            print(f"[index_builder] Crawled {len(urls)} URLs under {root}")
            crawl_urls.extend(urls)
        except Exception as exc:
            print(f"[index_builder] Failed to crawl {root}: {exc}")

    all_urls = list(set(FIXED_URLS + crawl_urls))
    print(f"[index_builder] Total URLs to load: {len(all_urls)}")
    for url in all_urls:
        print(f"  - {url}")

    records = load_web_docs(all_urls)
    print(f"[index_builder] Loaded {len(records)} raw section records")
    if not records:
        print("[index_builder] WARNING: No documents loaded; aborting index build.")
        return 0, []

    nodes = split_docs(records)
    docs = [Document(text=node["text"], metadata=node["metadata"]) for node in nodes]

    embeddings = get_embeddings()
    Settings.embed_model = embeddings
    sample_embedding = embeddings.get_text_embedding("dimension probe")
    faiss_index = faiss.IndexFlatL2(len(sample_embedding))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embeddings,
        show_progress=False,
    )
    index.set_index_id(INDEX_ID)

    Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=VECTORSTORE_PATH)
    _persist_node_cache(nodes)

    print(
        f"[index_builder] Saved LlamaIndex FAISS index to {VECTORSTORE_PATH} "
        f"(sections={len(nodes)})"
    )
    return len(nodes), nodes


def load_vectorstore():
    """Load the persisted LlamaIndex vector index from disk."""
    if not Path(VECTORSTORE_PATH).exists():
        raise FileNotFoundError(f"Persisted index not found at {VECTORSTORE_PATH}")

    embeddings = get_embeddings()
    Settings.embed_model = embeddings
    vector_store = FaissVectorStore.from_persist_dir(VECTORSTORE_PATH)
    storage_context = StorageContext.from_defaults(
        persist_dir=VECTORSTORE_PATH,
        vector_store=vector_store,
    )
    return load_index_from_storage(
        storage_context,
        index_id=INDEX_ID,
        embed_model=embeddings,
    )

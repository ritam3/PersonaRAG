# rag_core/index_builder.py

import re
from typing import List

import requests
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .embeddings_model import get_embeddings
from .config import VECTORSTORE_PATH
from .sources import CRAWL_ROOTS, FIXED_URLS
from .crawler import crawl_subpages


def _is_gdrive_file(url: str) -> bool:
    """Return True if this looks like a Google Drive file view URL."""
    return "drive.google.com" in url and "/file/d/" in url


def _gdrive_view_to_download(url: str) -> str:
    """
    Convert a Google Drive view URL to a direct download URL.

    Example:
      https://drive.google.com/file/d/<ID>/view
      -> https://drive.google.com/uc?export=download&id=<ID>
    """
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        return url
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _infer_section_label_from_url(url: str) -> str:
    """
    Heuristic: guess a section label from the URL path.
    e.g.
      https://your-site.com/about                 -> 'about'
      https://your-site.com/experience/juniper   -> 'experience/juniper'
    """
    try:
        path = url.split("://", 1)[-1].split("/", 1)[-1]
    except Exception:
        return url
    path = path.strip("/")
    if not path:
        return "root"
    return path


def _normalize_label(text: str, max_len: int = 80) -> str:
    """
    Create a short normalized label from a header text.
    - lowercases, removes newlines, collapses whitespace
    - trims to max_len and replaces spaces with '-' for compact labels
    """
    if not text:
        return ""
    lab = " ".join(text.split())  # collapse whitespace/newlines
    lab = lab.strip().lower()
    # shorten if too long
    if len(lab) > max_len:
        lab = lab[: max_len - 3].rstrip() + "..."
    # use a compact label form (slug-like) but keep readability
    slug = lab.replace(" ", "-")
    # remove characters that would be awkward in metadata keys
    slug = re.sub(r"[^a-z0-9_\-\.]+", "", slug)
    return slug


def load_web_docs(urls: List[str]):
    """
    Load documents from a list of URLs.

    - HTML URLs:
        * Fetch raw HTML with `requests`
        * Split into sections with HTMLSectionSplitter (h1/h2)
        * Infer section labels from the actual section header lines (preferred)
    - PDF URLs (including Google Drive file links):
        * Load with OnlinePDFLoader (one doc per page)
    """
    html_urls: List[str] = []
    pdf_urls: List[str] = []

    for url in urls:
        u = url.strip()
        if not u:
            continue

        # Google Drive file links (view) -> direct download PDF
        if _is_gdrive_file(u):
            pdf_urls.append(_gdrive_view_to_download(u))
        # Direct PDF URLs
        elif u.lower().endswith(".pdf"):
            pdf_urls.append(u)
        # Everything else is treated as HTML
        else:
            html_urls.append(u)

    docs = []

    # --- HTML via HTMLSectionSplitter on raw HTML string ---
    if html_urls:
        print(f"[index_builder] Loading HTML for {len(html_urls)} URLs with HTMLSectionSplitter")

        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ]
        html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

        for url in html_urls:
            try:
                print(f"[index_builder]   Fetching HTML from {url}")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                html_string = resp.text

                # HTMLSectionSplitter returns a list[Document] where each Document starts with the header line
                html_header_splits = html_splitter.split_text(html_string)

                # we will prefer to use the header line as the canonical label for each section
                # but ensure we deduplicate very similar headers within the same page
                seen_labels_in_page = set()

                print(f"[index_builder]   {url}: {len(html_header_splits)} HTML sections")

                for d in html_header_splits:
                    # source URL
                    d.metadata["source"] = url
                    # extract the first meaningful non-empty line as the header
                    first_lines = [ln.strip() for ln in d.page_content.splitlines() if ln.strip()]
                    header_line = first_lines[0] if first_lines else ""

                    # normalize header_line for metadata and label
                    # keep section_header as human readable short header (truncated if necessary)
                    human_header = header_line
                    if len(human_header) > 300:
                        human_header = human_header[:300] + "..."

                    if human_header != "More Project":
                        d.metadata["section_header"] = human_header
                    else:
                        # Extract endpoint from URL as fallback header
                        try:
                            # e.g. https://site.com/projects/my-cool-project -> "my-cool-project"
                            endpoint = url.rstrip("/").split("/")[-1]
                            # Make it human-readable
                            endpoint = endpoint.replace("-", " ").replace("_", " ").title()
                            d.metadata["section_header"] = endpoint
                            d.metadata["section_label"] = endpoint
                        except Exception:
                            # absolute fallback
                            d.metadata["section_header"] = human_header
                            d.metadata["section_label"] = human_header

                    # produce a short machine-friendly label from header; fallback to URL-based label
                    # label_from_header = _normalize_label(header_line)
                    # if not label_from_header:
                    #     label_from_header = _infer_section_label_from_url(url)

                    # # dedupe labels within this page (if splitter produced repeated headers)
                    # dedup_label = label_from_header
                    # suffix = 1
                    # while dedup_label in seen_labels_in_page:
                    #     dedup_label = f"{label_from_header}-{suffix}"
                    #     suffix += 1
                    # seen_labels_in_page.add(dedup_label)

                    # d.metadata["section_label"] = dedup_label
                    d.metadata["section_type"] = "remote_html"
                # append docs
                docs.extend(html_header_splits)
            except Exception as e:
                print(f"[index_builder] Error processing HTML from {url}: {e}")

    # --- PDF files (including Drive) via OnlinePDFLoader ---
    for pdf_url in pdf_urls:
        print(f"[index_builder] Loading PDF from {pdf_url}")
        try:
            pdf_loader = OnlinePDFLoader(pdf_url)
            pdf_docs = pdf_loader.load()
            section_label = _infer_section_label_from_url(pdf_url)
            for d in pdf_docs:
                d.metadata["source"] = pdf_url
                d.metadata["section_label"] = section_label
                d.metadata["section_type"] = "remote_pdf"
            docs.extend(pdf_docs)
            print(f"[index_builder]   {pdf_url}: {len(pdf_docs)} PDF pages")
        except Exception as e:
            print(f"[index_builder] Failed to load PDF from {pdf_url}: {e}")

    return docs


def split_docs(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split loaded documents into chunks for embedding.

    - HTML docs (from HTMLSectionSplitter) are already section-level chunks → keep as-is.
    - Non-HTML docs (PDF pages, etc.) are split with RecursiveCharacterTextSplitter.
    """
    html_docs = [d for d in docs if d.metadata.get("section_type") == "remote_html"]
    other_docs = [d for d in docs if d.metadata.get("section_type") != "remote_html"]

    chunks: List = []

    # Keep HTML sections as they are
    chunks.extend(html_docs)

    # Split other docs (PDFs, etc.) into text chunks
    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        other_chunks = splitter.split_documents(other_docs)
        for c in other_chunks:
            c.metadata.setdefault("section_label", c.metadata.get("source", "unknown"))
            c.metadata.setdefault(
                "section_type",
                c.metadata.get("section_type", "remote_pdf"),
            )
        chunks.extend(other_chunks)

    print(
        f"[index_builder] split_docs: {len(html_docs)} HTML section chunks, "
        f"{len(chunks) - len(html_docs)} non-HTML chunks"
    )
    return chunks


def build_and_save_index():
    """
    Crawl URLs, load docs, split into chunks, build FAISS index, and save it.

    Returns:
        (int, list): number of chunks indexed, and the chunks themselves.
    """
    # 1. Crawl project roots (and any other CRAWL_ROOTS) to get sub-URLs
    crawl_urls: List[str] = []
    for root in CRAWL_ROOTS:
        try:
            urls = crawl_subpages(root)
            print(f"[index_builder] Crawled {len(urls)} URLs under {root}")
            crawl_urls.extend(urls)
        except Exception as e:
            print(f"[index_builder] Failed to crawl {root}: {e}")

    # 2. Combine fixed URLs (resume, about, scholar, GitHub, etc.) + crawled URLs
    all_urls = list(set(FIXED_URLS + crawl_urls))

    print(f"[index_builder] Total URLs to load: {len(all_urls)}")
    for u in all_urls:
        print(f"  - {u}")

    docs = load_web_docs(all_urls)
    print(f"[index_builder] Loaded {len(docs)} raw documents")

    if not docs:
        print("[index_builder] WARNING: No documents loaded; aborting index build.")
        return 0, []

    # 3. Split into chunks (HTML via HTMLSectionSplitter, PDFs via recursive splitter)
    chunks = split_docs(docs)
    print(f"[index_builder] Split into {len(chunks)} chunks")

    if not chunks:
        print("[index_builder] WARNING: No chunks produced; aborting FAISS build.")
        return 0, []

    # 4. Build vector store with HF embeddings (e.g., all-MiniLM)
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)

    # 5. Save to disk
    vs.save_local(VECTORSTORE_PATH)
    print(
        f"[index_builder] Saved FAISS index to {VECTORSTORE_PATH} "
        f"(chunks={len(chunks)})"
    )

    return len(chunks), chunks


def load_vectorstore():
    """
    Load the FAISS vector store from disk using the same embedding model.
    """
    embeddings = get_embeddings()
    vs = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vs

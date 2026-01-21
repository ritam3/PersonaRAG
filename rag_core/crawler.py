# rag_core/crawler.py
from urllib.parse import urljoin, urlparse
from collections import deque
import requests
from bs4 import BeautifulSoup

def crawl_subpages(root_url: str, max_pages: int = 30, max_depth: int = 1):
    """
    Crawl subpages under the same path as root_url.
    E.g., root_url = https://site.com/projects/
    Will collect https://site.com/projects/* links.
    """
    parsed_root = urlparse(root_url)
    base_domain = parsed_root.netloc
    base_path = parsed_root.path.rstrip("/")  # "/projects"

    visited = set()
    to_visit = deque([(root_url, 0)])
    collected = set([root_url])

    while to_visit and len(collected) < max_pages:
        url, depth = to_visit.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            continue

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)

            # Stay in same domain
            if parsed.netloc != base_domain:
                continue

            # Stay under the same base path
            if not parsed.path.startswith(base_path):
                continue

            if full_url not in collected:
                collected.add(full_url)
                to_visit.append((full_url, depth + 1))

        visited.add(url)

    return list(collected)

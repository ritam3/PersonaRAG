# rag_core/sources.py

# Resume URL (HTML or PDF, both okay; PDF will be text-extracted)
LINKEDIN_URL = "https://www.linkedin.com/in/ritam-upadhyay-51ba81192/"

# Root projects page that links to each project subpage
PROJECTS_ROOT_URL = "https://fearless-writers-028990.framer.app/project"

# Other URLs directly relevant to your career
OTHER_CAREER_URLS = [
    "https://fearless-writers-028990.framer.app/old-home",
    "https://fearless-writers-028990.framer.app/",
    "https://scholar.google.com/citations?user=04o0bdcAAAAJ&hl=en",
    "https://fearless-writers-028990.framer.app/stack",
]

# These roots will be crawled:
CRAWL_ROOTS = [
   PROJECTS_ROOT_URL,
]

# These are direct, non-crawling URLs:
FIXED_URLS = [
    *OTHER_CAREER_URLS,
]

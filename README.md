---
title: PersonaRag
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.10.13"
app_file: app.py
pinned: false
---

# PersonaRag

PersonaRag is a production-oriented personal RAG assistant that turns a portfolio website, project pages, and career materials into a conversational digital twin. It is designed to answer recruiter, hiring manager, and collaborator questions about Ritam's experience, projects, research, education, and technical depth in a grounded, retrieval-backed way.

The system is not a generic chatbot wrapped around an LLM. It is a retrieval pipeline built to do three things well:
- understand career-focused user questions, including follow-ups and shorthand references
- retrieve the most relevant evidence from portfolio content and project pages
- answer in first person while staying tied to the indexed source material

## What This Project Does

PersonaRag lets a visitor ask questions such as:
- "What kind of ML work have you done?"
- "How does your experience align with prompt engineering roles?"
- "Tell me about your latest professional experience."
- "Explain the improper face one as well."
- "What projects demonstrate strong retrieval or LLM system design?"

It is intended to act like a smart front door to a candidate's body of work. Instead of forcing a recruiter to click through multiple project pages, decode resume bullets, or infer technical depth from scattered artifacts, PersonaRag assembles the most relevant context and responds directly.

## Why It Exists

A strong portfolio often contains rich information, but it is spread across:
- landing pages
- project detail pages
- experience summaries
- research and education sections
- external links and documents

A recruiter usually has limited time. PersonaRag compresses that exploration into a guided Q&A experience. It helps surface:
- relevant projects for a given role
- evidence of specific technical capabilities
- the connection between work experience and project execution
- the story behind systems, experiments, and implementation choices

## Core Features

- Gradio-based conversational interface for fast deployment and lightweight usage
- Website-first indexing over portfolio and project pages
- FAISS vector store for local retrieval
- Hybrid retrieval strategy that combines planner-directed scoping with label-aware routing
- LLM-based retrieval planning over real indexed page titles and summaries
- Follow-up question rewriting with stronger preservation of project aliases and shorthand references
- Evaluator pass that checks grounding and answer quality
- Single retry loop that revises weak answers instead of blindly returning first-pass output
- Daily refresh support so the assistant can stay aligned with website changes

## End-to-End System Flow

### 1. Content Discovery and Indexing

At indexing time, the system crawls configured portfolio roots and combines them with a fixed set of directly relevant URLs.

The current indexing pipeline:
- crawls project pages from the configured root
- fetches HTML content from the website
- splits pages into section-level chunks using HTML headers
- loads PDFs when present
- creates dense embeddings with a Hugging Face sentence-transformer
- stores the resulting chunks in a FAISS index

During indexing, the system also enriches metadata for downstream planning:
- `page_title`: a cleaned title-level header for the page
- `page_summary`: a 1-2 sentence heuristic summary extracted from the page content
- `planner_include`: whether this page should appear in the planner catalog

This is important because the planner no longer sees noisy repeated structural headers like `Project Overview` or `Key Highlights` as independent planning targets. Instead, it sees cleaner page-level entries such as:
- `PersonaRAG - A Conversation-Aware, Self-Correcting AI Twin`
- `Improper Face Detection At Frontend`
- `Work experience`
- `Research Experience`

### 2. Question Rewriting

For follow-up questions, the system rewrites the query into a standalone form. This step is intentionally conservative.

The rewrite layer now prioritizes:
- preserving project names and aliases
- preserving shorthand references that are useful for retrieval
- using history only to add missing context
- avoiding speculative paraphrases that could erase the best lexical match

For example, a follow-up like:
`Explain the improper face one as well`

should stay close to that wording rather than being transformed into an over-interpreted semantic description that weakens title matching.

### 3. Retrieval Planning

After rewrite, PersonaRag uses an LLM-based retrieval planner.

The planner receives:
- the user question
- a catalog of indexed page titles
- a lightweight summary for each included page

It then produces a structured retrieval plan that can include:
- a single focused retrieval step
- multiple section-targeted retrieval steps
- dependent retrieval flows where a later step builds on an earlier one

This planning layer is especially important for questions that are:
- multi-part
- comparative
- referential
- project-specific but phrased casually

Examples:
- "How do you align with prompt engineering roles?"
- "Tell me about your latest experience and the project behind it."
- "Explain the improper face one as well."

### 4. Scoped Retrieval

Once a plan is created, the retriever executes it step by step.

The retrieval behavior is intentionally staged:
- if the planner identifies explicit page titles, retrieval scopes directly to those pages
- if the planner does not provide a reliable scope, the system falls back to label-aware routing
- if label routing is weak, vector retrieval acts as a fallback

When a page title is selected, the system does not stop at the title chunk. It expands retrieval to sibling sections from the same page, prioritizing:
1. `Project Overview`
2. `Key Highlights`
3. title / identifying chunks

This is a critical design choice. It allows the system to use the title header to locate the correct page, then use the descriptive page content to answer in detail.

### 5. Grounded Answer Generation

The retrieved documents are passed into the answer chain with a system prompt that instructs the model to:
- answer in first person as Ritam
- stay grounded in retrieved context
- avoid unsupported claims
- refuse unrelated generic questions

The answer model is responsible for synthesizing the retrieved evidence into a concise, recruiter-friendly response.

### 6. Evaluation and Retry

After the first answer is produced, the system runs an evaluator model that checks:
- whether the answer follows the system instructions
- whether it actually answers the user question
- how grounded it is in the retrieved context
- whether there are signs of hallucination or low-quality response behavior

If the answer is weak, PersonaRag performs one revision pass. The revision prompt includes:
- the original question
- the first answer
- evaluator feedback
- strict grounding rules

This self-correction loop makes the assistant noticeably more robust than a single-pass RAG call.

## Example Scenarios PersonaRag Handles

### Recruiter Screening

A recruiter may ask:
- "What kinds of projects make you a fit for prompt engineering?"
- "What LLM-related work have you actually built?"
- "Do you have production-oriented AI experience?"

PersonaRag can retrieve project-level evidence and translate that into a direct fit narrative.

### Hiring Manager Deep Dive

A hiring manager may ask:
- "Tell me about a project where you built retrieval or evaluation pipelines."
- "How do your projects demonstrate practical machine learning ability?"
- "What technical judgment did you apply in your system design?"

The assistant can move from title-level identification to descriptive detail and explain what was built, why it mattered, and how it maps to the role.

### Follow-Up Navigation

A user may start broad and then drill down:
- "What projects are most relevant for this role?"
- "Explain the PersonaRAG one in more detail."
- "What about the improper face one?"

The system is designed to preserve those follow-up references and route them to the right project page instead of treating them as brand-new unrelated searches.

## Architecture Overview

### Interface

- `app.py`: lightweight application entry point
- `rag_core/ui.py`: Gradio UI construction
- `rag_core/runtime.py`: request lifecycle, refresh handling, answer generation, evaluation, and retry orchestration

### Retrieval and Planning

- `rag_core/rag_chain.py`: planner-aware retrieval and chain assembly
- `rag_core/planner.py`: header catalog building, retrieval planning, and entity extraction
- `rag_core/planner_schema.py`: structured plan and extraction schemas
- `rag_core/rag_chain_helper.py`: conversational rewrite logic

### Indexing and Data Preparation

- `rag_core/index_builder.py`: crawling, loading, metadata enrichment, chunking, embedding, and FAISS persistence
- `rag_core/crawler.py`: project-page crawling
- `rag_core/sources.py`: crawl roots and fixed URLs
- `rag_core/embeddings_model.py`: embedding model configuration

### Quality Control

- `rag_core/evaluator.py`: answer evaluator
- `rag_core/evaluator_schema.py`: evaluator schema
- `rag_core/logging_utils.py`: structured logging utilities

## Retrieval Philosophy

PersonaRag is built around a practical assumption: users often ask good questions in messy ways.

They may:
- refer to a project informally
- ask for a capability rather than a project name
- compare multiple parts of a career at once
- follow up with shorthand like "that one" or "the Paytm project"

Because of that, the system does not rely on a single brittle matching method. It combines:
- conversational rewriting
- planner-guided page selection
- scoped retrieval over page-level summaries and titles
- fallbacks when planning or matching is imperfect
- evaluator-driven revision

This makes the assistant more useful in real conversations, not just benchmark-style prompt formats.

## Logging and Traceability

The project includes structured logs that make it possible to inspect:
- the rewritten question
- the retrieval plan
- selected headers or scoped pages
- retrieved sources
- answer quality evaluation
- retry behavior when applicable

This is useful both for debugging and for improving retrieval quality over time.

## Deployment

This repository is configured to work with Hugging Face Spaces. The YAML front matter at the top of this file is required by Spaces and must remain at the top of the README.

There is also a GitHub Actions workflow that syncs a cleaned snapshot of the project to Hugging Face while stripping local vectorstore binaries from deployment artifacts.

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

By default, the app can rebuild the website index on startup. This behavior is controlled through environment variables such as:
- `REFRESH_ENABLED`
- `REFRESH_AT_HOUR`
- `REFRESH_AT_MINUTE`
- `REFRESH_ON_STARTUP`

## Why This Project Stands Out

PersonaRag demonstrates more than prompt usage. It shows practical system thinking across:
- retrieval design
- metadata-aware indexing
- LLM planning
- grounded answer generation
- answer evaluation and self-correction
- deployment-aware application structure

For a recruiter or hiring manager, the real value of PersonaRag is that it reflects both technical implementation skill and product judgment. It is a portfolio surface, a retrieval system, and an applied LLM engineering project packaged into one coherent experience.

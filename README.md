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

Live App: https://huggingface.co/spaces/ritup3/PersonaRag

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
- Starter-question dropdown for common recruiter and hiring-manager prompts
- Website-first indexing over portfolio and project pages
- LlamaIndex-based indexing and retrieval pipeline
- FAISS vector store used through LlamaIndex for local retrieval
- Tool-based routing over strict career sections such as `About`, `Work experience`, `Research Experience`, `Education`, and project pages
- Tool-description embedding router with exact intent overrides for common question types
- Hybrid retrieval inside each tool using BM25, vector search, and lexical scoring
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
- creates dense embeddings with a Hugging Face sentence-transformer through LlamaIndex
- stores the resulting chunks in a LlamaIndex-managed FAISS index

During indexing, the system also enriches metadata used later during retrieval:
- `page_title`: the resolved page title
- `page_type`: the type of source such as `project_detail`, `projects_index`, or `landing`
- `project_name`: normalized project identity for project detail pages
- `section_type`: semantic section label such as `experience`, `research`, `education`, or `projects`
- `section_label`: page-aware section label for logging and ranking
- `page_description`, `section_description`, and `chunk_description`: lightweight summaries used by retrieval and debugging

This metadata is primarily used for:
- defining strict tool boundaries
- improving within-tool retrieval
- making logs and retrieved sources easier to inspect

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

### 3. Tool Routing

After rewrite, PersonaRag chooses which retrieval tool should handle the question.

The current router is intentionally structured in two stages:
1. exact overrides for obvious intents
2. semantic routing using embeddings over hand-written tool behavior descriptions

The exact overrides strongly prefer the right tool when the question is explicit. Examples:
- self-introduction questions prefer `about`
- company / role / work-history questions prefer `experience`
- research / paper / publication questions prefer `research`
- education questions prefer `education`
- broad portfolio or multi-project questions prefer `projects`
- exact named project questions prefer `project_detail`

If the question is not explicit enough for an override, the router compares:
- the user query embedding
- the embedding of each tool’s behavior description

This keeps tool choice grounded in intended behavior rather than accidental overlap in page content.

The active tools are:
- `about`: only the `About` introduction section
- `experience`: only the `Work experience` section
- `research`: only the `Research Experience` section
- `education`: only the `Education` section
- `projects`: broad portfolio / multi-project questions
- `project_detail`: a specific named project page

### 4. Retrieval Inside the Chosen Tool

Once a tool is chosen, retrieval happens only inside that tool’s document set.

Within a tool, PersonaRag uses a hybrid document ranker:
- BM25 for strong lexical matching
- LlamaIndex vector retrieval over the FAISS index for semantic matching
- lexical boosts from section labels and metadata

This means routing answers the question “which section of the portfolio should handle this?”, and retrieval answers the question “which exact chunks inside that section should be shown to the model?”

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

- `rag_core/rag_chain.py`: tool definitions, tool-description router, clarification handling, and hybrid retrieval built on top of LlamaIndex retrievers
- `rag_core/rag_chain_helper.py`: conversational rewrite logic

### Indexing and Data Preparation

- `rag_core/index_builder.py`: crawling, loading, metadata enrichment, chunking, LlamaIndex document creation, embedding, and FAISS persistence
- `rag_core/crawler.py`: project-page crawling
- `rag_core/sources.py`: crawl roots and fixed URLs
- `rag_core/embeddings_model.py`: LlamaIndex embedding model configuration

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
- intent-aware tool routing
- hybrid retrieval inside the selected tool
- clarification when a broad summary and a more specific tool are both plausible
- evaluator-driven revision

This makes the assistant more useful in real conversations, not just benchmark-style prompt formats.

## Logging and Traceability

The project includes structured logs that make it possible to inspect:
- the rewritten question
- ranked tools from the router
- selected tool or clarification path
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
- intent-aware routing
- grounded answer generation
- answer evaluation and self-correction
- deployment-aware application structure

For a recruiter or hiring manager, the real value of PersonaRag is that it reflects both technical implementation skill and product judgment. It is a portfolio surface, a retrieval system, and an applied LLM engineering project packaged into one coherent experience.

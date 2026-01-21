# rag_core/rag_chain.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag_core.FusedRetreiver import FusedRetriever
from rag_core.PrefixRetreiver import PrefixRetriever


from .models_groq import get_answer_llm, get_judge_llm

def faiss_all_docs(faiss_store):
    # Map FAISS index positions -> docstore ids
    ids = list(faiss_store.index_to_docstore_id.values())
    # Pull Documents from the docstore in that order
    return [faiss_store.docstore.search(_id) for _id in ids]


SYSTEM_PROMPT = """
You are a QA assistant for Ritam that answers questions about Ritam's career posing as Ritam's digital twin.

You will receive:
- `context`: text chunks retrieved from Ritam's resume, website, project pages, scholar profile, etc.
- `chat_history`: a short summary of the prior conversation turns for coherence.

Context:
{context}

Chat history (for reference):
{chat_history}

Rules:
- Use ONLY the provided context for factual claims.
- If the answer is not clearly supported by the context, say "I don't know"
  and suggest rephrasing in terms of Ritam's work, projects, or research.
- If the user asks generic questions unrelated to Ritam, politely refuse saying that please ask only career related questions to me.

The chat_history is only for understanding the follow-up question; do NOT invent facts
that aren't in the context even if they appear earlier in chat_history.

"""

def build_rag_chain(vectorstore, k=2, max_lines=3, weights=(0.35, 0),top_k=None):
    """
    Returns (rag_chain, retriever, SYSTEM_PROMPT)

    - vectorstore: your FAISS/Chroma/… already loaded
    - k: per-retriever fetch size (each retriever will pull up to k docs)
    - max_lines: how many leading lines to consider for prefix matching
    - weights: contribution of (Prefix, Vector) retrievers in the fusion
    - top_k: final number of docs returned by the ensemble (defaults to k)
    """
    # ---- Retrievers
    all_docs = faiss_all_docs(vectorstore)
    prefix_retriever = PrefixRetriever(docs=all_docs, k=2, max_lines=3)

    # Pull a few extra from vectorstore to give the fusion more to work with
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": max(k, 1)})

    fused_retriever = FusedRetriever(
                        prefix_retriever=prefix_retriever,
                        vector_retriever=vector_retriever,
                        k=6,
                        prefix_first=True,
                    )

    # ---- LLM + prompt
    llm = get_answer_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # ---- Stuff docs into the prompt, then make the RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(fused_retriever, document_chain)

    return rag_chain, fused_retriever, SYSTEM_PROMPT



# ---------- Conversational question rewriter ----------

# We’ll use a low-temperature model (judge_llm) to rewrite follow-up questions
_rewriter_llm = get_judge_llm()

QUESTION_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a helpful assistant that rewrites follow-up questions into standalone questions.

You are given a chat history between a user and an assistant, plus the user's new question.
Your job is to rewrite the new question so that it is self-contained and can be understood
without the previous turns.

The rewritten question MUST stay faithful to the user's intent and be about Ritam's
career, projects, research, or education.
If the question is already standalone, return it as-is.
"""),
        ("human", """
Chat history:
------------
{chat_history}

New user question:
------------
{question}

Rewrite the new question into a single, self-contained question:
"""),
    ]
)

def rewrite_question_with_history(history, question: str) -> str:
    """
    history: list of [user_msg, assistant_msg] pairs (from Gradio ChatInterface)
    question: current user string

    Returns a standalone question string that incorporates context from history.
    """
    # If no history, just return the question
    if not history:
        return question

    # Convert history to a single text block
    history_lines = []
    for turn in history:
        if not turn or len(turn) < 2:
            continue
        user_msg, assistant_msg = turn[0], turn[1]
        history_lines.append(f"User: {user_msg}")
        history_lines.append(f"Assistant: {assistant_msg}")
    history_text = "\n".join(history_lines)

    chain = QUESTION_REWRITE_PROMPT | _rewriter_llm
    resp = chain.invoke({"chat_history": history_text, "question": question})
    standalone = resp.content.strip()
    return standalone or question

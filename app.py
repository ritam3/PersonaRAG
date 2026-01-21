# app.py
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import time
import threading
import atexit

from rag_core.index_builder import load_vectorstore
from rag_core.rag_chain_helper import rewrite_question_with_history
from rag_core.rag_chain import build_rag_chain
from rag_core.evaluator import evaluate_answer
from rag_core.index_builder import build_and_save_index

from rag_core.config import VECTORSTORE_PATH

# ---------- Refresh config ----------
REFRESH_ENABLED = os.getenv("REFRESH_ENABLED", "true").lower() == "true"
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", str(24 * 60 * 60)))
REFRESH_AT_HOUR = int(os.getenv("REFRESH_AT_HOUR", "3"))
REFRESH_AT_MINUTE = int(os.getenv("REFRESH_AT_MINUTE", "0"))
REFRESH_ONLY_FIXED_URLS = os.getenv("REFRESH_ONLY_FIXED_URLS", "false").lower() == "true"

state_lock = threading.RLock()
stop_refresh_event = threading.Event()


load_dotenv()

# ---------- Logging (Model Flow) ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("model_flow")
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    h.setFormatter(formatter)
    logger.addHandler(h)

def log_event(event: str, **payload):
    """Structured-ish logging for tracing model flow."""
    safe = {}
    for k, v in payload.items():
        try:
            json.dumps(v)  # ensure serializable
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    logger.info("%s | %s", event, json.dumps(safe, ensure_ascii=False))


# ---------- Global state ----------
vectorstore = None
rag_chain = None
retriever = None
system_prompt = None


def init_rag():
    global vectorstore, rag_chain, retriever, system_prompt

    # HARD DISABLE: no crawling / no auto-index build
    index_path = Path(VECTORSTORE_PATH) / "index.faiss"
    if not index_path.exists():
        n_chunks, _ = build_and_save_index()
        log_event("refresh.index_built", mode="crawl", chunks=n_chunks)

    vectorstore = load_vectorstore()

    rag_chain, retriever, system_prompt = build_rag_chain(
        vectorstore,
        k=5,
        max_docs=2
    )
    log_event("init_rag.ready", vectorstore_path=VECTORSTORE_PATH)


init_rag()

def refresh_rag_once():
    """
    Refetch website docs and rebuild the index + chain.
    Never crashes the app; logs errors.
    """
    global vectorstore, rag_chain, retriever, system_prompt

    log_event("refresh.start", only_fixed_urls=REFRESH_ONLY_FIXED_URLS)

    try:
        from rag_core.rag_chain import build_rag_chain
        from rag_core.index_builder import load_vectorstore

        n_chunks, _ = build_and_save_index()
        log_event("refresh.index_built", mode="crawl", chunks=n_chunks)

        # Reload from disk (ensures consistent serialization)
        vs = load_vectorstore()

        # Build new chain
        new_chain, new_retriever, new_system_prompt = build_rag_chain(
            vs,
            k=5,
            max_docs=2,
        )

        # Atomic swap
        with state_lock:
            vectorstore = vs
            rag_chain = new_chain
            retriever = new_retriever
            system_prompt = new_system_prompt

        log_event("refresh.done", status="ok")

    except Exception as e:
        log_event("refresh.error", error=str(e))

def _seconds_until_next_run(hour: int, minute: int) -> int:
    # compute sleep until next local time (hour:minute)
    now = time.localtime()
    target = time.mktime((
        now.tm_year, now.tm_mon, now.tm_mday,
        hour, minute, 0,
        now.tm_wday, now.tm_yday, now.tm_isdst
    ))
    now_ts = time.time()
    if target <= now_ts:
        target += 24 * 60 * 60
    return int(target - now_ts)

def _daily_refresh_loop():
    # small startup delay
    time.sleep(3)

    while not stop_refresh_event.is_set():
        # sleep until next scheduled time
        sleep_s = _seconds_until_next_run(REFRESH_AT_HOUR, REFRESH_AT_MINUTE)
        log_event("refresh.sleep", seconds=sleep_s, at_hour=REFRESH_AT_HOUR, at_minute=REFRESH_AT_MINUTE)

        # sleep in chunks so shutdown responds quickly
        while sleep_s > 0 and not stop_refresh_event.is_set():
            step = min(5, sleep_s)
            time.sleep(step)
            sleep_s -= step

        if stop_refresh_event.is_set():
            break

        refresh_rag_once()

def start_refresh_thread():
    if not REFRESH_ENABLED:
        log_event("refresh.disabled")
        return
    t = threading.Thread(target=_daily_refresh_loop, daemon=True)
    t.start()
    log_event("refresh.thread_started", daily_at=f"{REFRESH_AT_HOUR:02d}:{REFRESH_AT_MINUTE:02d}")

atexit.register(lambda: stop_refresh_event.set())
start_refresh_thread()



# ---------- Helpers ----------
def _history_to_text(history):
    """Convert Gradio history ([[user, bot], ...]) to a readable text snippet."""
    if not history:
        return ""
    lines = []
    for turn in history:
        if not turn or len(turn) < 2:
            continue
        user_msg, assistant_msg = turn[0], turn[1]
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


def _docs_to_loggable(docs, max_chars=220):
    """Return lightweight doc info for logs (no full dump)."""
    out = []
    for d in (docs or []):
        src = (d.metadata or {}).get("source", "unknown")
        txt = (d.page_content or "").strip().replace("\n", " ")
        out.append({
            "source": src,
            "preview": (txt[:max_chars] + ("..." if len(txt) > max_chars else "")),
            "metadata": (d.metadata or {}),
        })
    return out


def generate_answer(message, history):
    """
    Core logic:
    - rewrite question with history (best-effort)
    - run RAG (required; if this fails, return a fallback reply)
    - evaluate (best-effort; if this fails, skip retry)
    - optionally retry once based on evaluator signal
    Returns ONLY the final answer string (no sources/context/evaluator in UI).
    """
    log_event("request.start", user_message=message)

    # ---------- 1. Rewrite with history (best-effort) ----------
    try:
        standalone_question = rewrite_question_with_history(history, message)
    except Exception as e:
        log_event("rewrite.error", error=str(e))
        standalone_question = message  # fallback: use original message

    history_text = _history_to_text(history)

    log_event(
        "rewrite.done",
        standalone_question=standalone_question,
        history_chars=len(history_text),
    )

    # ---------- 2. Run RAG (if this fails, we bail with generic error text) ----------
    try:
        with state_lock:
            local_rag_chain = rag_chain
            local_system_prompt = system_prompt

        rag_res = local_rag_chain.invoke({
            "input": standalone_question,
            "chat_history": history_text,
        })
    except Exception as e:
        log_event("rag.error", error=str(e))
        fallback = (
            "I'm having trouble accessing my knowledge base right now. "
            "Please try again in a moment."
        )
        log_event(
            "request.end",
            final_answer_preview=fallback[:400] + ("..." if len(fallback) > 400 else "")
        )
        return fallback

    answer_1 = rag_res.get("answer", "") or ""
    ctx_docs_1 = rag_res.get("context", []) or []

    log_event(
        "rag.done",
        answer_preview=answer_1[:400] + ("..." if len(answer_1) > 400 else ""),
        retrieved_count=len(ctx_docs_1),
        retrieved_docs=_docs_to_loggable(ctx_docs_1),
    )

    # ---------- 3. Evaluate (best-effort; never crash on judge failure) ----------
    eval_res_1 = None
    try:
        eval_res_1 = evaluate_answer(
            system_prompt=local_system_prompt,
            question=message,
            context_docs=ctx_docs_1,
            answer=answer_1,
        )

        log_event(
            "eval.done",
            overall_score=float(eval_res_1.overall_score),
            grounded=float(eval_res_1.grounded_in_context_score),
            hallucination=bool(eval_res_1.hallucination_detected),
            feedback=str(eval_res_1.feedback),
        )
    except Exception as e:
        log_event("eval.error", error=str(e))
        # We just skip retry logic; answer_1 is still valid.

    final_answer = answer_1

    # ---------- 4. Single retry (only if evaluator succeeded & says to retry) ----------
    try:
        if (
            eval_res_1 is not None and
           ( eval_res_1.overall_score < 0.70 or getattr(eval_res_1, "should_retry", True))
        ):
            revision_prompt = (
                f"{standalone_question}\n\n"
                f"You previously answered this:\n{answer_1}\n\n"
                "An evaluator found issues. Revise your answer to address the feedback below.\n"
                "Rules:\n"
                "- Use ONLY the provided context.\n"
                "- If the context does not support the claim, say \"I don't know\".\n"
                "- Be specific and grounded.\n\n"
                f"Evaluator feedback:\n{eval_res_1.feedback}\n"
            )

            log_event(
                "retry.triggered",
                reason="eval_score_below_threshold",
                threshold=0.90,
            )

            # RAG retry — if this fails, we keep the original answer_1
            try:
                rag_res_2 = rag_chain.invoke({
                    "input": revision_prompt,
                    "chat_history": history_text,
                })
                answer_2 = rag_res_2.get("answer", "") or ""
                ctx_docs_2 = rag_res_2.get("context", []) or []

                log_event(
                    "rag.retry_done",
                    answer_preview=answer_2[:400] + ("..." if len(answer_2) > 400 else ""),
                    retrieved_count=len(ctx_docs_2),
                    retrieved_docs=_docs_to_loggable(ctx_docs_2),
                )

                # Optional: re-evaluate the revised answer (ignore errors)
                try:
                    eval_res_2 = evaluate_answer(
                        system_prompt=system_prompt,
                        question=message,
                        context_docs=ctx_docs_2,
                        answer=answer_2,
                    )
                    log_event(
                        "eval.retry_done",
                        overall_score=float(eval_res_2.overall_score),
                        grounded=float(eval_res_2.grounded_in_context_score),
                        hallucination=bool(eval_res_2.hallucination_detected),
                        feedback=str(eval_res_2.feedback),
                    )
                except Exception as e_eval2:
                    log_event("eval.retry_error", error=str(e_eval2))

                # If we got here, second answer is safe to use
                if eval_res_1.overall_score>eval_res_2.overall_score:
                    final_answer = answer_1
                else:
                    final_answer = answer_2

            except Exception as e_rag2:
                # Retry RAG failed; log and fall back to first answer
                log_event("rag.retry_error", error=str(e_rag2))
                final_answer = answer_1

    except Exception as e_retry_block:
        # Any unexpected error in retry logic should not crash the whole request
        log_event("retry.block_error", error=str(e_retry_block))
        final_answer = answer_1

    # ---------- 5. Final logging & return ----------
    log_event(
        "request.end",
        final_answer_preview=final_answer[:400] + ("..." if len(final_answer) > 400 else "")
    )

    return final_answer


def respond(message, history):
    """
    Gradio wrapper that is resilient to unexpected exceptions.
    If anything explodes inside generate_answer, we log it and return
    a safe fallback message.
    """
    if not message:
        return "", history

    try:
        answer = generate_answer(message, history)
    except Exception as e:
        log_event("respond.fatal_error", error=str(e))
        answer = (
            "Something went wrong on my side while trying to answer. "
            "Please try again in a moment."
        )

    history = history + [[message, answer]]
    return "", history


# ---------- Gradio UI ----------
with gr.Blocks(title="Ask Ritam (Career QA Bot)") as demo:
    gr.Markdown(
        "# Ask Ritam\n"
        "A RAG-powered career assistant over my resume, website, and projects.\n"
        "Ask anything about my experience, projects, research, or education."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask anything about my career, projects, or research...",
                    lines=2,
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear chat")

            send_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch()

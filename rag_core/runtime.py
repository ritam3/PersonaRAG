import atexit
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from rag_core.config import NODE_CACHE_PATH, VECTORSTORE_PATH
from rag_core.evaluator import evaluate_answer
from rag_core.index_builder import build_and_save_index, load_node_cache, load_vectorstore
from rag_core.logging_utils import get_model_flow_logger, log_event
from rag_core.rag_chain import ContextDocument, build_rag_chain
from rag_core.rag_chain_helper import rewrite_question_with_history


@dataclass(frozen=True)
class RefreshConfig:
    enabled: bool
    at_hour: int
    at_minute: int
    only_fixed_urls: bool
    rebuild_on_startup: bool

    @classmethod
    def from_env(cls) -> "RefreshConfig":
        return cls(
            enabled=os.getenv("REFRESH_ENABLED", "true").lower() == "true",
            at_hour=int(os.getenv("REFRESH_AT_HOUR", "3")),
            at_minute=int(os.getenv("REFRESH_AT_MINUTE", "0")),
            only_fixed_urls=os.getenv("REFRESH_ONLY_FIXED_URLS", "false").lower() == "true",
            rebuild_on_startup=os.getenv("REFRESH_ON_STARTUP", "false").lower() == "true",
        )


def _history_to_text(history: Sequence[Sequence[str]]) -> str:
    """Convert Gradio history ([[user, bot], ...]) into a compact text block."""
    if not history:
        return ""

    lines: List[str] = []
    for turn in history:
        if not turn or len(turn) < 2:
            continue
        user_msg, assistant_msg = turn[0], turn[1]
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


def _docs_to_loggable(docs: Sequence[Any], max_chars: int = 220) -> List[dict]:
    """Return lightweight document metadata for logs without dumping full context."""
    summaries: List[dict] = []
    for doc in docs or []:
        source = (doc.metadata or {}).get("source", "unknown")
        text = (doc.page_content or "").strip().replace("\n", " ")
        summaries.append(
            {
                "source": source,
                "preview": text[:max_chars] + ("..." if len(text) > max_chars else ""),
                "metadata": doc.metadata or {},
            }
        )
    return summaries


class CareerQARuntime:
    """Owns model state, refresh scheduling, and answer generation."""

    def __init__(self, refresh_config: RefreshConfig | None = None):
        self.refresh_config = refresh_config or RefreshConfig.from_env()
        self.logger = get_model_flow_logger()
        self.state_lock = threading.RLock()
        self.stop_refresh_event = threading.Event()
        self.vectorstore = None
        self.rag_chain = None
        self.retriever = None
        self.system_prompt = None
        self.refresh_thread = None
        self.pending_clarification: Dict[str, Any] | None = None

        self.init_rag()
        self.start_refresh_thread()
        atexit.register(self.stop)

    def _load_context_docs(self) -> List[ContextDocument]:
        return [
            ContextDocument(
                page_content=record["text"],
                metadata=record["metadata"],
            )
            for record in load_node_cache()
        ]

    def init_rag(self) -> None:
        """Build the index if needed, then load the vectorstore and chain."""
        index_path = NODE_CACHE_PATH

        should_rebuild = self.refresh_config.rebuild_on_startup or not index_path.exists()
        if should_rebuild:
            try:
                chunk_count, _ = build_and_save_index()
                self.log_event(
                    "refresh.index_built",
                    mode="startup_rebuild",
                    chunks=chunk_count,
                )
            except Exception as exc:
                if not index_path.exists():
                    raise
                self.log_event(
                    "refresh.startup_rebuild_failed",
                    error=str(exc),
                    fallback="loading_existing_index",
                )

        vector_index = load_vectorstore()
        docs = self._load_context_docs()
        rag_chain, retriever, system_prompt = build_rag_chain(
            vector_index,
            docs,
            k=5,
            max_docs=3,
        )

        with self.state_lock:
            self.vectorstore = vector_index
            self.rag_chain = rag_chain
            self.retriever = retriever
            self.system_prompt = system_prompt

        self.log_event("init_rag.ready", vectorstore_path=VECTORSTORE_PATH)

    def log_event(self, event: str, **payload) -> None:
        log_event(self.logger, event, **payload)

    def refresh_rag_once(self) -> None:
        """Rebuild the index and atomically swap in a fresh chain."""
        self.log_event(
            "refresh.start",
            only_fixed_urls=self.refresh_config.only_fixed_urls,
        )

        try:
            chunk_count, _ = build_and_save_index()
            self.log_event("refresh.index_built", mode="crawl", chunks=chunk_count)

            vector_index = load_vectorstore()
            docs = self._load_context_docs()
            rag_chain, retriever, system_prompt = build_rag_chain(
                vector_index,
                docs,
                k=5,
                max_docs=3,
            )

            with self.state_lock:
                self.vectorstore = vector_index
                self.rag_chain = rag_chain
                self.retriever = retriever
                self.system_prompt = system_prompt

            self.log_event("refresh.done", status="ok")
        except Exception as exc:
            self.log_event("refresh.error", error=str(exc))

    def _seconds_until_next_run(self) -> int:
        """Compute the delay until the next scheduled refresh in local time."""
        now = time.localtime()
        target = time.mktime(
            (
                now.tm_year,
                now.tm_mon,
                now.tm_mday,
                self.refresh_config.at_hour,
                self.refresh_config.at_minute,
                0,
                now.tm_wday,
                now.tm_yday,
                now.tm_isdst,
            )
        )
        now_ts = time.time()
        if target <= now_ts:
            target += 24 * 60 * 60
        return int(target - now_ts)

    def _daily_refresh_loop(self) -> None:
        time.sleep(3)

        while not self.stop_refresh_event.is_set():
            sleep_seconds = self._seconds_until_next_run()
            self.log_event(
                "refresh.sleep",
                seconds=sleep_seconds,
                at_hour=self.refresh_config.at_hour,
                at_minute=self.refresh_config.at_minute,
            )

            while sleep_seconds > 0 and not self.stop_refresh_event.is_set():
                step = min(5, sleep_seconds)
                time.sleep(step)
                sleep_seconds -= step

            if self.stop_refresh_event.is_set():
                break

            self.refresh_rag_once()

    def start_refresh_thread(self) -> None:
        if not self.refresh_config.enabled:
            self.log_event("refresh.disabled")
            return

        if self.refresh_thread and self.refresh_thread.is_alive():
            return

        self.refresh_thread = threading.Thread(
            target=self._daily_refresh_loop,
            daemon=True,
        )
        self.refresh_thread.start()
        self.log_event(
            "refresh.thread_started",
            daily_at=f"{self.refresh_config.at_hour:02d}:{self.refresh_config.at_minute:02d}",
        )

    def stop(self) -> None:
        self.stop_refresh_event.set()

    def _run_rag(
        self,
        question: str,
        history_text: str,
        forced_tool: str | None = None,
    ) -> Dict[str, Any]:
        with self.state_lock:
            local_rag_chain = self.rag_chain

        payload: Dict[str, Any] = {
            "input": question,
            "chat_history": history_text,
        }
        if forced_tool:
            payload["forced_tool"] = forced_tool
        return local_rag_chain.invoke(payload)

    def generate_answer(self, message: str, history: Sequence[Sequence[str]]) -> str:
        """Run rewrite, RAG, evaluation, and optional retry for one user message."""
        self.log_event("request.start", user_message=message)
        history_text = _history_to_text(history)

        with self.state_lock:
            pending = self.pending_clarification

        if pending:
            with self.state_lock:
                local_rag_chain = self.rag_chain
            forced_tool = local_rag_chain.resolve_clarification_reply(
                message,
                pending.get("candidate_tools", []),
                pending.get("preferred_tool", "about"),
            )
            standalone_question = pending.get("original_question", message)
            self.log_event(
                "routing.clarification_resolved",
                original_question=standalone_question,
                clarification_reply=message,
                forced_tool=forced_tool,
                candidate_tools=pending.get("candidate_tools", []),
            )
            with self.state_lock:
                self.pending_clarification = None
            try:
                rag_result = self._run_rag(standalone_question, history_text, forced_tool=forced_tool)
            except Exception as exc:
                self.log_event("rag.error", error=str(exc))
                fallback = (
                    "I'm having trouble accessing my knowledge base right now. "
                    "Please try again in a moment."
                )
                self.log_event("request.end", final_answer_preview=fallback[:400])
                return fallback
        else:
            try:
                standalone_question = rewrite_question_with_history(history, message)
            except Exception as exc:
                self.log_event("rewrite.error", error=str(exc))
                standalone_question = message

            self.log_event(
                "rewrite.done",
                standalone_question=standalone_question,
                history_chars=len(history_text),
            )

            try:
                rag_result = self._run_rag(standalone_question, history_text)
            except Exception as exc:
                self.log_event("rag.error", error=str(exc))
                fallback = (
                    "I'm having trouble accessing my knowledge base right now. "
                    "Please try again in a moment."
                )
                self.log_event("request.end", final_answer_preview=fallback[:400])
                return fallback

        if rag_result.get("needs_clarification"):
            clarification_answer = rag_result.get("answer", "") or (
                "Could you clarify which area you want me to focus on?"
            )
            with self.state_lock:
                self.pending_clarification = {
                    "candidate_tools": rag_result.get("candidate_tools", []),
                    "preferred_tool": rag_result.get("preferred_tool", "about"),
                    "original_question": rag_result.get("original_question", standalone_question),
                }
            self.log_event(
                "routing.clarification_requested",
                original_question=standalone_question,
                candidate_tools=rag_result.get("candidate_tools", []),
                preferred_tool=rag_result.get("preferred_tool", "about"),
            )
            self.log_event("request.end", final_answer_preview=clarification_answer[:400])
            return clarification_answer

        answer_1 = rag_result.get("answer", "") or ""
        context_docs_1 = rag_result.get("context", []) or []

        self.log_event(
            "rag.done",
            answer_preview=answer_1[:400] + ("..." if len(answer_1) > 400 else ""),
            retrieved_count=len(context_docs_1),
            retrieved_docs=_docs_to_loggable(context_docs_1),
        )

        with self.state_lock:
            local_system_prompt = self.system_prompt

        eval_result_1 = None
        try:
            eval_result_1 = evaluate_answer(
                system_prompt=local_system_prompt,
                question=message,
                context_docs=context_docs_1,
                answer=answer_1,
            )
            self.log_event(
                "eval.done",
                overall_score=float(eval_result_1.overall_score),
                grounded=float(eval_result_1.grounded_in_context_score),
                hallucination=bool(eval_result_1.hallucination_detected),
                feedback=str(eval_result_1.feedback),
            )
        except Exception as exc:
            self.log_event("eval.error", error=str(exc))

        final_answer = answer_1
        try:
            should_retry = (
                eval_result_1 is not None
                and (
                    eval_result_1.overall_score < 0.70
                    or getattr(eval_result_1, "should_retry", True)
                )
            )
            if should_retry:
                revision_prompt = (
                    f"{standalone_question}\n\n"
                    f"You previously answered this:\n{answer_1}\n\n"
                    "An evaluator found issues. Revise your answer to address the feedback below.\n"
                    "Rules:\n"
                    "- Use ONLY the provided context.\n"
                    '- If the context does not support the claim, say "I don\'t know".\n'
                    "- Be specific and grounded.\n\n"
                    f"Evaluator feedback:\n{eval_result_1.feedback}\n"
                )
                self.log_event(
                    "retry.triggered",
                    reason="eval_score_below_threshold",
                    threshold=0.90,
                )

                try:
                    retry_result = self._run_rag(revision_prompt, history_text)
                    answer_2 = retry_result.get("answer", "") or ""
                    context_docs_2 = retry_result.get("context", []) or []
                    self.log_event(
                        "rag.retry_done",
                        answer_preview=answer_2[:400] + ("..." if len(answer_2) > 400 else ""),
                        retrieved_count=len(context_docs_2),
                        retrieved_docs=_docs_to_loggable(context_docs_2),
                    )

                    eval_result_2 = None
                    try:
                        eval_result_2 = evaluate_answer(
                            system_prompt=local_system_prompt,
                            question=message,
                            context_docs=context_docs_2,
                            answer=answer_2,
                        )
                        self.log_event(
                            "eval.retry_done",
                            overall_score=float(eval_result_2.overall_score),
                            grounded=float(eval_result_2.grounded_in_context_score),
                            hallucination=bool(eval_result_2.hallucination_detected),
                            feedback=str(eval_result_2.feedback),
                        )
                    except Exception as exc:
                        self.log_event("eval.retry_error", error=str(exc))

                    if eval_result_2 is not None and eval_result_1.overall_score <= eval_result_2.overall_score:
                        final_answer = answer_2
                    else:
                        final_answer = answer_1
                except Exception as exc:
                    self.log_event("rag.retry_error", error=str(exc))
                    final_answer = answer_1
        except Exception as exc:
            self.log_event("retry.block_error", error=str(exc))
            final_answer = answer_1

        self.log_event(
            "request.end",
            final_answer_preview=final_answer[:400] + ("..." if len(final_answer) > 400 else ""),
        )
        return final_answer

    def respond(self, message: str, history: Sequence[Sequence[str]]):
        """Gradio callback wrapper that converts failures into safe user responses."""
        history = history or []
        if not message:
            return "", history

        try:
            answer = self.generate_answer(message, history)
        except Exception as exc:
            self.log_event("respond.fatal_error", error=str(exc))
            answer = (
                "Something went wrong on my side while trying to answer. "
                "Please try again in a moment."
            )

        updated_history = list(history) + [[message, answer]]
        return "", updated_history

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import message_text
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    FactCandidate,
    UserProfileStore,
    compose_answer,
    estimate_tokens,
    extract_fact_candidates,
    is_question,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Dùng hồ sơ User.md bên dưới làm nguồn sự thật về người dùng; "
    "nếu có đính chính, luôn ưu tiên thông tin mới nhất. Chỉ lưu fact ổn định (tên, nơi ở, nghề, sở thích, style) "
    "bằng tool remember_user_fact, không lưu chuyện tạm thời hay câu đùa. Trả lời đúng style người dùng thích."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    Per turn:
    message → extract facts (confidence-gated) → upsert User.md → append to compact memory
    → prompt = system + User.md + summary + recent messages → reply → update counters.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            confidence_threshold=self.config.memory_confidence_threshold,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
            summary_max_items=self.config.summary_max_items,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.system_prompt_tokens = estimate_tokens(ADVANCED_SYSTEM_PROMPT)
        self._active_user: str | None = None
        self._active_thread: str | None = None
        self.langchain_agent = None if (force_offline or self.config.offline) else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # -- shared turn bookkeeping -------------------------------------------

    def _begin_turn(self, user_id: str, thread_id: str, message: str) -> tuple[list[str], int]:
        """Steps 1-4: extract → persist → append → account prompt load. Returns (changes, overhead tokens)."""

        changes = self.profile_store.apply_candidates(user_id, extract_fact_candidates(message))
        summary_before = self.compact_memory.context(thread_id)["summary_tokens_generated"]
        self.compact_memory.append(thread_id, "user", message)
        summary_after = self.compact_memory.context(thread_id)["summary_tokens_generated"]

        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + (
            self._estimate_prompt_context_tokens(user_id, thread_id)
        )
        # Memory work is real output: writing User.md lines and producing summaries cost tokens too.
        overhead = sum(estimate_tokens(change) for change in changes) + (summary_after - summary_before)
        return changes, overhead

    def _end_turn(self, thread_id: str, message: str, response: str, overhead: int) -> None:
        summary_before = self.compact_memory.context(thread_id)["summary_tokens_generated"]
        self.compact_memory.append(thread_id, "assistant", response)
        overhead += self.compact_memory.context(thread_id)["summary_tokens_generated"] - summary_before
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(message) + estimate_tokens(response) + overhead
        )

    def _result(self, thread_id: str, response: str, mode: str, changes: list[str]) -> dict[str, Any]:
        return {
            "response": response,
            "mode": mode,
            "memory_updates": changes,
            "agent_tokens": self.token_usage(thread_id),
            "prompt_tokens": self.prompt_token_usage(thread_id),
            "compactions": self.compaction_count(thread_id),
        }

    # -- offline path ------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changes, overhead = self._begin_turn(user_id, thread_id, message)
        response = self._offline_response(user_id, thread_id, message, changes)
        self._end_turn(thread_id, message, response, overhead)
        return self._result(thread_id, response, "offline", changes)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system + User.md (prompt view) + compact summary + kept recent messages."""

        thread = self.compact_memory.context(thread_id)
        return (
            self.system_prompt_tokens
            + estimate_tokens(self.profile_store.prompt_view(user_id))
            + estimate_tokens(thread["summary"])
            + sum(estimate_tokens(m["content"]) for m in thread["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str, changes: list[str] | None = None) -> str:
        """Deterministic answer that reads persisted memory (User.md), never the raw old threads."""

        if is_question(message):
            answer = compose_answer(message, self.profile_store.facts(user_id))
            if answer:
                return answer
        if changes:
            return "Đã cập nhật User.md: " + "; ".join(changes) + "."
        return "Đã ghi nhận."

    # -- live path ---------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changes, overhead = self._begin_turn(user_id, thread_id, message)
        self._active_user = user_id
        self._active_thread = thread_id
        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        response = message_text(result["messages"][-1])
        self._end_turn(thread_id, message, response, overhead)
        return self._result(thread_id, response, "live", changes)

    def _maybe_build_langchain_agent(self):
        """Live agent: tools over User.md + dynamic profile prompt + summarization middleware."""

        if not self.config.model.is_configured:
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        store = self.profile_store
        agent = self

        @tool
        def read_user_memory() -> str:
            """Đọc toàn bộ User.md (hồ sơ bền vững) của người dùng hiện tại."""

            return store.read_text(agent._active_user or "anonymous")

        @tool
        def remember_user_fact(key: str, value: str) -> str:
            """Lưu/đính chính một fact ổn định. key ∈ name, location, profession, drink, food, pet, response_style, interests, hobbies."""

            user_id = agent._active_user or "anonymous"
            # LLM-proposed facts go through the same confidence gate as extracted ones.
            applied = store.apply_candidates(user_id, [FactCandidate(key, value, 0.7, "llm tool")])
            return "saved: " + "; ".join(applied) if applied else "no change"

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            context = getattr(request.runtime, "context", None)
            user_id = getattr(context, "user_id", None) or agent._active_user or "anonymous"
            summary = agent.compact_memory.context(agent._active_thread or "")["summary"]
            extra = f"\n\nTóm tắt các lượt cũ:\n{summary}" if summary else ""
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n{store.prompt_view(user_id)}{extra}"

        model = build_chat_model(self.config.model)
        return create_agent(
            model,
            tools=[read_user_memory, remember_user_fact],
            middleware=[
                inject_profile,
                SummarizationMiddleware(
                    model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )

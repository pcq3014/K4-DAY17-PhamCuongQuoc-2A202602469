from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import compose_answer, estimate_tokens, extract_profile_updates, is_question
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Trả lời ngắn gọn dựa trên cuộc hội thoại hiện tại. "
    "Bạn không có bộ nhớ dài hạn: nếu thông tin không có trong thread này, hãy nói là chưa biết."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def message_text(message: Any) -> str:
    """Extract plain text from a LangChain message (content may be a list of blocks)."""

    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` and re-sends all of it every turn.
    - No `User.md`, no compaction: a new thread starts from zero.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.system_prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT)
        self.langchain_agent = None if (force_offline or self.config.offline) else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    # -- internals ----------------------------------------------------------

    def _record_user_turn(self, thread_id: str, message: str) -> SessionState:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        # The whole thread history is the prompt: this is exactly what grows without compaction.
        session.prompt_tokens_processed += self.system_prompt_tokens + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        return session

    def _record_assistant_turn(self, session: SessionState, message: str, response: str) -> None:
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._record_user_turn(thread_id, message)

        # Facts are only derived from *this* thread's messages, never from other threads.
        thread_facts: dict[str, str] = {}
        for past in session.messages[:-1]:
            if past["role"] == "user":
                thread_facts.update(extract_profile_updates(past["content"]))

        answer = compose_answer(message, thread_facts) if is_question(message) else None
        response = answer or "Đã ghi nhận trong phiên này."

        self._record_assistant_turn(session, message, response)
        return {
            "response": response,
            "mode": "offline",
            "agent_tokens": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._record_user_turn(thread_id, message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = message_text(result["messages"][-1])
        self._record_assistant_turn(session, message, response)
        return {
            "response": response,
            "mode": "live",
            "agent_tokens": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Wire `create_agent` + `InMemorySaver` (thread-scoped memory only); None if unavailable."""

        if not self.config.model.is_configured:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

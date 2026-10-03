from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, provider_config_from_env


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths for the repo root, dataset directory, and state directory
    - compact-memory settings (threshold + number of recent messages kept)
    - provider settings for the main model and the judge model
    - guardrail for persistent memory (confidence threshold before writing User.md)
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    memory_confidence_threshold: float = 0.6
    summary_max_items: int = 6
    offline: bool = True


def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:
        # Minimal fallback parser: KEY=VALUE lines, '#' comments.
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` (if any) and return a populated LabConfig.

    Env vars:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE
    - JUDGE_PROVIDER / JUDGE_MODEL (default: same as LLM_*)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - CUSTOM_BASE_URL / CUSTOM_API_KEY, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, MEMORY_CONFIDENCE_THRESHOLD
    - LAB_OFFLINE=1 forces the deterministic offline mode (default when no key is set)
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = Path(os.getenv("LAB_STATE_DIR", root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    model = provider_config_from_env("LLM")
    judge_model = provider_config_from_env("JUDGE", fallback=model)

    offline_env = os.getenv("LAB_OFFLINE")
    if offline_env is not None:
        offline = offline_env.strip().lower() in ("1", "true", "yes")
    else:
        offline = not (os.getenv("LLM_PROVIDER") and model.is_configured)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        # ~800 tokens: normal 10-turn chats never compact, the long stress thread compacts several times.
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        memory_confidence_threshold=float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.6")),
        summary_max_items=int(os.getenv("SUMMARY_MAX_ITEMS", "6")),
        offline=offline,
    )

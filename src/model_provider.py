from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common typos / alternative names students (and .env files) tend to use.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}

# Env var that holds the API key for each provider (ollama needs none).
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "qwen2.5:7b",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    @property
    def is_configured(self) -> bool:
        """True when the provider has what it needs to make a live call."""

        if self.provider == "ollama":
            return True
        if self.provider == "custom":
            return bool(self.base_url)
        return bool(self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; raise on unknown providers."""

    key = (value or "").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider '{value}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    return _PROVIDER_ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so offline mode works without any provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        elif provider == "custom":
            raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL.")
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
        except ImportError:
            # OpenRouter is OpenAI-compatible, so fall back to ChatOpenAI.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )

    raise ValueError(f"Unsupported provider: {provider}")


def provider_config_from_env(prefix: str = "LLM", fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<PREFIX>_PROVIDER`, `<PREFIX>_MODEL`, `<PREFIX>_TEMPERATURE` (+ provider key/base URL)."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv(f"{prefix}_MODEL") or (
        fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    )
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", fallback.temperature if fallback else 0.0))

    key_env = PROVIDER_KEY_ENV[provider]
    api_key = os.getenv(key_env) if key_env else None
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    base_url = {
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "ollama": os.getenv("OLLAMA_BASE_URL"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL"),
        "openai": os.getenv("OPENAI_BASE_URL"),
    }.get(provider)

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key or None,
        base_url=base_url or None,
    )

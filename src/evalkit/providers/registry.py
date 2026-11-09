"""Named providers, configured once per process and built on first use.

A judge or a prompt target names its provider: ``anthropic``, ``openai``,
``gemini``, or a server the operator described in ``EVALKIT_PROVIDERS_FILE``:

    [providers.local]
    kind = "openai_compat"
    base_url = "http://vllm.internal:8000/v1"
    api_key_env = "LOCAL_LLM_KEY"      # optional; a variable or LOCAL_LLM_KEY_FILE
    output_limit_field = "max_tokens"  # older servers
    reasoning_effort = false
    price_list = "openai"              # optional; a proxy in front of a priced API

Only configured names can be used, so a run config sent to the API cannot
point a judge at an arbitrary URL. ``openai`` and ``gemini`` are built in, with
their public endpoints and keys from ``OPENAI_API_KEY`` and ``GEMINI_API_KEY``;
the file can redefine them, for example to go through a proxy.

Each provider prices its models from its own table in the ``PriceBook``:
``anthropic``, ``openai`` and ``gemini`` start from their built-in lists, and a
server from the file starts from the list it names in ``price_list``, or from
none (``EVALKIT_PRICES_FILE`` then has to price the models it serves).

Each name gets one client for the life of the registry. The Claude client is
passed in (or made on first use by a factory) by whoever owns it, the CLI or
the API, since prompt targets use it too, and is not closed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from evalkit.errors import InvalidInputError
from evalkit.pricing import PRICE_LISTS, PriceBook, PriceTable
from evalkit.providers import gemini, openai_compat
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.base import Provider
from evalkit.targets.specs import ANTHROPIC, PROVIDER_NAME

__all__ = ["ANTHROPIC", "PROVIDER_NAME", "ProviderConfig", "ProviderRegistry", "builtin_providers"]


class ProviderConfig(BaseModel):
    """One HTTP provider: what kind of API it speaks, where, and with which key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["openai_compat", "gemini"]
    base_url: str = Field(min_length=1, max_length=2048)
    api_key: SecretStr | None = None
    # The variable the key comes from, named when a required key is missing.
    api_key_env: str | None = Field(default=None, max_length=128)
    # The public APIs refuse every request without a key; self-hosted servers often take none.
    key_required: bool = False
    output_limit_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    reasoning_effort: bool = False
    thinking_level: bool = False
    timeout_seconds: float = Field(default=600.0, gt=0, le=3600)
    # The built-in price list this server's models start from (default: the
    # provider's own, which only ``openai`` and ``gemini`` have).
    price_list: str | None = Field(default=None, max_length=64)

    @field_validator("price_list")
    @classmethod
    def _known_list(cls, value: str | None) -> str | None:
        if value is not None and value not in PRICE_LISTS:
            raise ValueError(f"unknown price list {value!r}; built in: {', '.join(PRICE_LISTS)}")
        return value

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        url = httpx2.URL(value)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("base_url must be an http or https URL")
        if url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must not carry credentials, a query or a fragment")
        return value.rstrip("/")

    @property
    def endpoint(self) -> str:
        """What the judge identity records: the kind of API and where it is."""
        return f"{self.kind}:{self.base_url}"


def builtin_providers(
    openai_key: SecretStr | None, gemini_key: SecretStr | None
) -> dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(
            kind="openai_compat",
            base_url=openai_compat.DEFAULT_BASE_URL,
            api_key=openai_key,
            api_key_env="OPENAI_API_KEY",
            key_required=True,
        ),
        "gemini": ProviderConfig(
            kind="gemini",
            base_url=gemini.DEFAULT_BASE_URL,
            api_key=gemini_key,
            api_key_env="GEMINI_API_KEY",
            key_required=True,
        ),
    }


class ProviderRegistry:
    def __init__(
        self,
        configs: Mapping[str, ProviderConfig],
        prices: PriceBook,
        *,
        claude: ClaudeClient | None = None,
        claude_factory: Callable[[], ClaudeClient] | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if ANTHROPIC in configs:
            raise InvalidInputError(f"the provider name {ANTHROPIC!r} is reserved for Claude")
        self.configs = dict(configs)
        self.prices = prices
        self.claude = claude
        self._claude_factory = claude_factory
        self._transport = transport
        self._clients: dict[str, Provider] = {}

    def names(self) -> list[str]:
        return sorted([ANTHROPIC, *self.configs])

    def endpoint(self, name: str) -> str | None:
        """Where ``name`` sends requests, or None for Claude (one fixed API)."""
        if name == ANTHROPIC:
            return None
        return self._config(name).endpoint

    def price_table(self, name: str) -> PriceTable:
        """The prices provider ``name`` charges by, without building its client."""
        if name == ANTHROPIC:
            if self.claude is not None:
                return self.claude.prices
            return self.prices.table(ANTHROPIC)
        return self.prices.table(name, self._config(name).price_list)

    def get(self, name: str) -> Provider:
        if name == ANTHROPIC:
            if self.claude is None and self._claude_factory is not None:
                self.claude = self._claude_factory()
            if self.claude is None:
                raise InvalidInputError("the Claude API is not configured; set ANTHROPIC_API_KEY")
            return self.claude
        client = self._clients.get(name)
        if client is None:
            client = self._build(name, self._config(name))
            self._clients[name] = client
        return client

    def _config(self, name: str) -> ProviderConfig:
        config = self.configs.get(name)
        if config is None:
            raise InvalidInputError(
                f"unknown provider {name!r}; configured: {', '.join(self.names())}"
            )
        return config

    def _build(self, name: str, config: ProviderConfig) -> Provider:
        # Say so before the run starts rather than as an auth error on every case.
        if config.api_key is None and config.key_required:
            raise InvalidInputError(f"provider {name} needs a key; set {config.api_key_env}")
        if config.kind == "gemini":
            return gemini.GeminiClient(
                self.price_table(name),
                config.api_key,
                name=name,
                base_url=config.base_url,
                thinking_level=config.thinking_level,
                timeout=config.timeout_seconds,
                transport=self._transport,
            )
        return openai_compat.OpenAICompatClient(
            self.price_table(name),
            config.api_key,
            name=name,
            base_url=config.base_url,
            output_limit_field=config.output_limit_field,
            reasoning_effort=config.reasoning_effort,
            timeout=config.timeout_seconds,
            transport=self._transport,
        )

    async def aclose(self) -> None:
        """Close the clients this registry built (not the Claude client it was given)."""
        clients, self._clients = self._clients, {}
        for client in clients.values():
            await client.aclose()

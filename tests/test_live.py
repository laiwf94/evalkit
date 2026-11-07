"""One real call to the Claude API. Runs only when ANTHROPIC_API_KEY is set:

ANTHROPIC_API_KEY=... uv run pytest -m live --no-cov
"""

import asyncio
import os

import pytest

from evalkit.config import Settings
from evalkit.datasets import Case
from evalkit.pricing import PriceTable
from evalkit.providers.anthropic import ClaudeClient
from evalkit.targets import PromptTargetSpec
from evalkit.targets.prompt import PromptTarget

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def test_prompt_target_against_the_real_api() -> None:
    settings = Settings.from_env()
    client = ClaudeClient(PriceTable(settings.prices), settings.anthropic_api_key)
    spec = PromptTargetSpec(
        type="prompt",
        template="What is {{a}} plus {{b}}? Reply with the number only.",
        max_tokens=256,
        effort="low",
    )
    target = PromptTarget(spec, client)

    async def main() -> None:
        try:
            gen = await target.generate(Case(id="sum", input={"a": 2, "b": 3}, expected="5"))
        finally:
            await client.aclose()
        assert "5" in gen.text
        assert gen.usage.input_tokens > 0
        assert gen.usage.cost_usd > 0

    asyncio.run(main())

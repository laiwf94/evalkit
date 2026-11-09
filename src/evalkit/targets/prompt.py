"""A prompt template evaluated on a model, on any provider."""

from __future__ import annotations

from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.providers.base import Provider
from evalkit.targets.base import Generation, Target
from evalkit.targets.specs import PromptTargetSpec
from evalkit.targets.template import render


class PromptTarget(Target):
    def __init__(self, spec: PromptTargetSpec, client: Provider) -> None:
        if spec.prompt is not None:
            raise InvalidInputError(
                f"prompt {spec.prompt!r} must be resolved from the registry before it runs"
            )
        # An unknown model would run with no cost accounting; refuse it up front.
        client.prices.require(spec.model)
        # Rate limits and cool-downs are shared per provider.
        self.provider = client.name
        self.spec = spec
        self.client = client

    def prompt_for(self, case: Case) -> str:
        return render(self.spec.template, case.input)

    async def generate(self, case: Case) -> Generation:
        spec = self.spec
        completion = await self.client.complete(
            model=spec.model,
            prompt=self.prompt_for(case),
            system=spec.system,
            max_tokens=spec.max_tokens,
            effort=spec.effort,
            fallbacks=spec.fallbacks,
        )
        return Generation(completion.text, completion.stop_reason, completion.usage)

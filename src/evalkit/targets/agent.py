"""A model working on each case with tools from a sandbox.

The case input fills the template, as in a prompt target, and the model then
runs the tool loop (``evalkit.tools.loop``) against a fresh ``ToolSandbox``
for the case: scripted and recorded answers, limits and injected faults, all
seeded by the case id. The output is the model's last reply; every reply and
every tool call comes back in ``Generation.steps`` for the ``result_steps``
table, so the whole trajectory is stored with the result.
"""

from __future__ import annotations

from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.providers.base import Provider
from evalkit.providers.tools import ToolDefinition, ToolProvider
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.targets.base import Generation, Target
from evalkit.targets.specs import AgentTargetSpec
from evalkit.targets.template import render
from evalkit.tools.loop import StepStore, ToolLoop, owner_hash
from evalkit.tools.sandbox import ToolSandbox


class AgentTarget(Target):
    # The run's outer limiter only paces cases; each model call takes its
    # provider's limiter (``attach``), as a prompt target's call would.
    provider = "agent"
    caches_cases = False

    def __init__(
        self, spec: AgentTargetSpec, client: Provider, *, endpoint: str | None = None
    ) -> None:
        if not isinstance(client, ToolProvider):
            raise InvalidInputError(f"provider {client.name} cannot call tools")
        # An unknown model would run with no cost accounting; refuse it up front.
        client.prices.require(spec.model)
        self.spec = spec
        self.client = client
        self.fixtures = spec.recorded()
        tools = [ToolDefinition.of(d) for d in spec.sandbox.definitions()]
        self.loop = ToolLoop(
            client,
            model=spec.model,
            tools=tools,
            owner_hash=owner_hash({"spec": spec.model_part(), "extra": endpoint}, tools),
            system=spec.system,
            max_tokens=spec.max_tokens,
            effort=spec.effort,
            fallbacks=spec.fallbacks,
            max_steps=spec.max_steps,
        )

    def attach(
        self,
        *,
        store: StepStore | None,
        redactor: Redactor,
        limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        self.loop.attach(
            store=store, redactor=redactor, limiter=limiter, max_retry_after=max_retry_after
        )

    async def generate(self, case: Case) -> Generation:
        task = render(self.spec.template, case.input)
        # A fresh sandbox per attempt: a retried case sees the same answers and
        # faults, so its transcript, and with it the step cache, is the same.
        sandbox = ToolSandbox(self.spec.sandbox, self.fixtures, session=case.id)
        result = await self.loop.run(task, sandbox)
        return Generation(
            text=result.text, stop_reason=result.stop_reason, usage=result.usage, steps=result.steps
        )

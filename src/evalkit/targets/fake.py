"""A target with known behaviour: answers correctly for a seeded fraction of cases."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict

from evalkit.datasets import Case
from evalkit.targets.base import Generation, Target, TargetError, Usage
from evalkit.targets.specs import FakeTargetSpec


def unit(*parts: object) -> float:
    """A uniform number in [0, 1) derived from ``parts``: the same input, the same number."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeTarget(Target):
    provider = "fake"

    def __init__(self, spec: FakeTargetSpec) -> None:
        self.spec = spec
        self._attempts: dict[str, int] = defaultdict(int)
        self.calls = 0

    async def generate(self, case: Case) -> Generation:
        spec = self.spec
        self.calls += 1
        attempt = self._attempts[case.id]
        self._attempts[case.id] += 1
        delay = spec.latency_ms + spec.jitter_ms * unit(spec.seed, case.id, attempt, "jitter")
        if delay:
            await asyncio.sleep(delay / 1000)
        if unit(spec.seed, case.id, attempt, "fail") < spec.failure_rate:
            raise TargetError(
                "fake transient failure", retryable=True, retry_after=spec.retry_after_s
            )
        if unit(spec.seed, case.id) < spec.accuracy and case.expected is not None:
            text = case.expected if isinstance(case.expected, str) else json.dumps(case.expected)
        else:
            text = spec.wrong_answer
        prompt = case.input if isinstance(case.input, str) else json.dumps(case.input)
        usage = Usage(input_tokens=max(1, len(prompt) // 4), output_tokens=max(1, len(text) // 4))
        return Generation(text=text, stop_reason="end_turn", usage=usage)

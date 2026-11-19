"""``semantic_similarity``: cosine similarity between the answer and the reference
under an embedding model.

Without a threshold the score is the cosine clipped to [0, 1], a smooth
measure for reports. With one it is 1 when the cosine reaches the threshold
and 0 otherwise, which is what a gate should read: raw cosines of unrelated
texts sit well above 0 for most models (often 0.6 to 0.8), so their mean moves
little when answers go wrong. ``evalkit calibrate --scorer`` picks the
threshold from human pass/fail labels (``calibrate_threshold`` in
``evalkit.scorers.calibration``).

The scorer records an identity like a judge's (provider, endpoint, model and,
as its rubric, the threshold and reference key), so the gate refuses to
compare two runs scored by different embedding models or thresholds.
"""

from __future__ import annotations

from typing import Any

from evalkit.datasets import Case, canonical_json
from evalkit.embeddings import Embedder, cosine
from evalkit.providers.base import Provider
from evalkit.scorers.base import Scorer, ScoreResult, expected_value
from evalkit.scorers.judge import judge_identity
from evalkit.targets.base import TargetError

MODE = "semantic_similarity"


class SemanticSimilarity(Scorer):
    def __init__(
        self,
        name: str,
        *,
        embedder: Embedder,
        threshold: float | None = None,
        expected_key: str | None = None,
    ) -> None:
        super().__init__(name)
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a cosine, between -1 and 1")
        self.embedder = embedder
        self.threshold = threshold
        self.expected_key = expected_key

    # What the runner reads to price and identify the scorer, as for a judge.
    @property
    def provider(self) -> str:
        return self.embedder.provider

    @property
    def client(self) -> Provider:
        return self.embedder.client

    @property
    def model(self) -> str:
        return self.embedder.model

    def identity(self) -> dict[str, Any]:
        settings = {"threshold": self.threshold, "expected_key": self.expected_key}
        return judge_identity(
            provider=self.embedder.provider,
            endpoint=self.embedder.endpoint,
            model=self.embedder.model,
            mode=MODE,
            effort="none",
            rubric=canonical_json(settings),
        )

    def reference(self, case: Case) -> str | None:
        value = expected_value(case, self.expected_key)
        return value if isinstance(value, str) and value.strip() else None

    async def similarity(self, case: Case, output: str) -> tuple[float | None, ScoreResult]:
        """The cosine (None when there is none to give) and the unthresholded
        result: what calibration sweeps over. Raises ``TargetError`` when the
        embedding request fails after its retries."""
        reference = self.reference(case)
        if reference is None:
            return None, ScoreResult.unscorable("no reference text to compare with")
        if not output.strip():
            return 0.0, ScoreResult(0.0, "empty output")
        got = await self.embedder.embed([output, reference])
        usage = got.usage
        value = cosine(got.vectors[0], got.vectors[1])
        if value is None:
            return None, ScoreResult(
                None, "a zero vector has no direction", usage.input_tokens, 0, usage.cost_usd
            )
        result = ScoreResult(
            max(0.0, value), f"cosine {value:.4f}", usage.input_tokens, 0, usage.cost_usd
        )
        return value, result

    async def score(self, case: Case, output: str) -> ScoreResult:
        try:
            value, result = await self.similarity(case, output)
        except TargetError as exc:
            return ScoreResult.unscorable(f"embedding call failed: {exc.message}")
        if value is None or self.threshold is None:
            return result
        verdict = ">=" if value >= self.threshold else "<"
        return ScoreResult(
            1.0 if value >= self.threshold else 0.0,
            f"cosine {value:.4f} {verdict} {self.threshold}",
            result.input_tokens,
            0,
            result.cost_usd,
        )

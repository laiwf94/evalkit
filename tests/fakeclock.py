"""Virtual time for tests: sleeping advances the clock instead of waiting."""

import asyncio


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        target = self.now + seconds
        await asyncio.sleep(0)
        self.now = max(self.now, target)

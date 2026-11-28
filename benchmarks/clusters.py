"""How long failure clustering takes, and whether it finds planted groups.

    uv run python benchmarks/clusters.py                          # 5,000 cases, 1536 dims
    uv run python benchmarks/clusters.py --cases 20000 --groups 8

Plants ``--groups`` groups of unit vectors (a random centre per group, noise
about it) the size of an embedding model's output, then times
``evalkit.clusters.fit_clusters`` with k chosen by silhouette, which is all the
work the report does besides the embedding and naming calls. Prints the time,
the k it chose, the purity (the share of cases in their group's majority
cluster) and the peak memory of the arrays. Exits 1 if k or the groups are
not recovered, or if it takes longer than ``--max-seconds``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

from evalkit.clusters import ClusterSpec, fit_clusters, normalize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=5_000)
    parser.add_argument("--dims", type=int, default=1536)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.6, help="noise length per vector")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    centers = normalize(rng.normal(size=(args.groups, args.dims)))
    truth = rng.integers(args.groups, size=args.cases)
    noise = rng.normal(scale=args.noise / math.sqrt(args.dims), size=(args.cases, args.dims))
    vectors = list((centers[truth] + noise).astype(np.float32))
    tracemalloc.start()
    start = time.perf_counter()
    fit = fit_clusters(vectors, ClusterSpec(max_k=10))
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    labels = np.asarray(fit.labels)
    purity = (
        sum(Counter(labels[truth == g].tolist()).most_common(1)[0][1] for g in range(args.groups))
        / args.cases
    )
    print(
        f"{args.cases} cases x {args.dims} dims, {args.groups} groups: {elapsed:.2f} s, "
        f"k {fit.choice.k} ({fit.choice.by}, silhouette {fit.choice.silhouette or 0:.3f}), "
        f"purity {purity:.4f}, peak {peak / 2**20:.0f} MiB"
    )
    ok = fit.choice.k == args.groups and purity == 1.0 and elapsed <= args.max_seconds
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

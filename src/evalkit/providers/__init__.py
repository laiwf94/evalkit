"""Model providers behind one protocol: see ``evalkit.providers.base``."""

from evalkit.providers.base import (
    Completion,
    Embeddings,
    ErrorKind,
    Provider,
    ProviderError,
    kind_for_status,
)

__all__ = [
    "Completion",
    "Embeddings",
    "ErrorKind",
    "Provider",
    "ProviderError",
    "kind_for_status",
]

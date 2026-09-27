"""Optional semantic retrieval adapters; importing this module never uses a network.

A local model can implement EmbeddingProvider. The OpenRouter adapter is only
used when explicitly constructed and passed to MemoryStore. Its requests send
query text and eligible memory content to OpenRouter and may incur API charges.
API contract: https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
from threading import Lock
from typing import Protocol, Sequence
from urllib import request


class EmbeddingProvider(Protocol):
    """Provider identity must change whenever model or vector settings change."""

    @property
    def fingerprint(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vectors(vectors: object, expected_count: int) -> list[list[float]]:
    """Reject corrupt, zero-length or incompatible embeddings before caching."""
    if not isinstance(vectors, (list, tuple)) or len(vectors) != expected_count:
        raise ValueError("Embedding provider returned an unexpected vector count")
    result: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("Embedding vectors must be nonempty lists")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
            raise ValueError("Embedding vectors must contain numbers")
        clean = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in clean) or not any(clean):
            raise ValueError("Embedding vectors must be finite and nonzero")
        if dimension is not None and len(clean) != dimension:
            raise ValueError("Embedding dimensions do not match")
        dimension = len(clean)
        result.append(clean)
    return result


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if not left_norm or not right_norm or not math.isfinite(left_norm * right_norm):
        raise ValueError("Embedding norms must be finite and nonzero")
    return max(-1.0, min(1.0, sum((a / left_norm) * (b / right_norm) for a, b in zip(left, right))))


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """Validate once and retain unit vectors for inexpensive repeated comparisons."""
    clean = validate_vectors([vector], 1)[0]
    norm = math.hypot(*clean)
    if not math.isfinite(norm):
        raise ValueError("Embedding norm must be finite")
    return tuple(value / norm for value in clean)


class VectorCache:
    """Process-local LRU bounded by both entry count and vector coordinates.

    Keys are model/content hashes, never raw query text. Coordinates are also
    bounded because providers can return vectors of very different dimensions.
    """

    def __init__(self, *, max_entries: int, max_coordinates: int) -> None:
        self.max_entries = max_entries
        self.max_coordinates = max_coordinates
        self._values: OrderedDict[tuple[str, ...], tuple[float, ...]] = OrderedDict()
        self._coordinates = 0
        self._lock = Lock()

    def get(self, key: tuple[str, ...]) -> tuple[float, ...] | None:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key: tuple[str, ...], vector: tuple[float, ...]) -> None:
        if len(vector) > self.max_coordinates:
            return
        with self._lock:
            previous = self._values.pop(key, None)
            self._coordinates -= len(previous) if previous is not None else 0
            self._values[key] = vector
            self._coordinates += len(vector)
            while len(self._values) > self.max_entries or self._coordinates > self.max_coordinates:
                _, removed = self._values.popitem(last=False)
                self._coordinates -= len(removed)


@dataclass(frozen=True)
class OpenRouterEmbeddingProvider:
    """Explicit opt-in HTTPS provider; credentials are never included in repr/cache."""

    api_key: str = field(repr=False)
    model: str = "openai/text-embedding-3-small"
    timeout: float = 15.0
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if not self.api_key.strip() or not self.model.strip():
            raise ValueError("Embedding API key and model are required")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Embedding timeout must be positive and finite")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")

    @property
    def fingerprint(self) -> str:
        return f"openrouter:v1:{self.model}:dimensions={self.dimensions or 'default'}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        http_request = request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            result = json.load(response)
        data = result.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding response has an invalid data field")
        if any(not isinstance(item, dict) or type(item.get("index")) is not int for item in data):
            raise ValueError("Embedding response has invalid indexes")
        ordered = sorted(data, key=lambda item: item["index"])
        if [item["index"] for item in ordered] != list(range(len(texts))):
            raise ValueError("Embedding response has missing or duplicate indexes")
        return validate_vectors([item.get("embedding") for item in ordered], len(texts))

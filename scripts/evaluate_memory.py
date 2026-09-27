"""Offline-by-default retrieval evaluation; never accesses user memory.

Run: python -m scripts.evaluate_memory --output Documentation/memory-evaluation-results.json
The synthetic fixture is a development benchmark, not the assignment's unseen
end-to-end coding-task evaluation. Baselines operate on memory records as proxies
for context selection; they do not measure an LLM's answer quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
from time import perf_counter
from typing import Any, Callable

from memory import MemoryRecord, MemoryStore
from memory.embeddings import EmbeddingProvider, OpenRouterEmbeddingProvider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "memory_retrieval.json"
SCOPE_FIELDS = ("project_id", "user_id", "session_id", "task_id", "agent_id")


def estimated_tokens(content: str) -> int:
    """Match the application's heuristic, not model-specific billed tokens."""
    return math.ceil(len(content.encode("utf-8")) / 3)


class TrackedEmbeddingProvider:
    """Expose provider failures instead of mistaking fallback for hybrid evidence."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.calls = 0
        self.input_texts = 0
        self.failures: Counter[str] = Counter()

    @property
    def fingerprint(self) -> str:
        return self.provider.fingerprint

    def embed(self, texts: Any) -> list[list[float]]:
        self.calls += 1
        self.input_texts += len(texts)
        try:
            return self.provider.embed(texts)
        except Exception as error:
            self.failures[type(error).__name__] += 1
            raise


def render_record(record: MemoryRecord) -> str:
    """Use the same source-bearing representation for every non-empty baseline."""
    return json.dumps(
        {"id": record.id, "type": record.memory_type,
         "source": record.source_reference, "content": record.content},
        ensure_ascii=False, sort_keys=True,
    )


def visible_and_active(record: MemoryRecord, scope: dict[str, Any]) -> bool:
    """Independent access oracle used for the proxy and isolation measurement."""
    if record.status != "active":
        return False
    if record.valid_until and datetime.fromisoformat(record.valid_until) <= datetime.now(timezone.utc):
        return False
    include_global = scope.get("include_global", True)
    for field in SCOPE_FIELDS:
        stored, requested = getattr(record, field), scope.get(field)
        if requested is None:
            if stored is not None:
                return False
        elif stored != requested and not (include_global and stored is None):
            return False
    return True


def load_fixture(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported fixture schema")
    keys = [entry["key"] for entry in data["memories"]]
    ids = [case["id"] for case in data["cases"]]
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError("Memory keys and query IDs must be unique")
    if len(ids) < 30:
        raise ValueError("Benchmark requires at least 30 retrieval queries")
    for case in data["cases"]:
        expected, forbidden = set(case["expected"]), set(case.get("forbidden", []))
        if expected & forbidden or not (expected | forbidden) <= set(keys):
            raise ValueError(f"Invalid relevance labels for {case['id']}")
    return data


def seed_store(store: MemoryStore, data: dict[str, Any]) -> dict[str, MemoryRecord]:
    records: dict[str, MemoryRecord] = {}
    for entry in data["memories"]:
        scope = {**data["default_scope"], **entry.get("scope", {})}
        kwargs = {
            **scope, "source_type": "document",
            "source_reference": f"memory_retrieval.json#{entry['key']}",
            "reliability": entry.get("reliability", 0.9), "importance": 0.6,
        }
        if "valid_until" in entry:
            kwargs["valid_until"] = entry["valid_until"]
        if "supersedes" in entry:
            record = store.supersede(records[entry["supersedes"]].id, entry["content"], **kwargs)
        else:
            record = store.add(entry["content"], entry["memory_type"], **kwargs)
        records[entry["key"]] = record
        if entry.get("update_status"):
            store.update(record.id, status=entry["update_status"])
        if entry.get("delete"):
            store.delete(record.id)
    # Refresh statuses changed by updates/supersession. Keep the deleted ID so
    # a forbidden result remains traceable if a backend wrongly returns it.
    return {key: store.get(record.id) or record for key, record in records.items()}


def bound_context(records: list[MemoryRecord], limit: int, budget: int) -> list[MemoryRecord]:
    selected: list[MemoryRecord] = []
    used = 0
    for record in records:
        cost = estimated_tokens(render_record(record)) + (1 if selected else 0)
        if used + cost <= budget:
            selected.append(record)
            used += cost
        if len(selected) == limit:
            break
    return selected


def query_metrics(returned: list[str], expected: list[str], limit: int) -> dict[str, float | bool | None]:
    relevant = set(expected)
    found = len(set(returned) & relevant)
    reciprocal = next((1.0 / rank for rank, key in enumerate(returned, 1) if key in relevant), 0.0)
    return {
        "precision": found / len(returned) if relevant and returned else (0.0 if relevant else None),
        "precision_at_k": found / limit if relevant else None,
        "recall": found / len(relevant) if relevant else None,
        "reciprocal_rank": reciprocal if relevant else None,
        "exact_set_match": set(returned) == relevant,
        "correct_abstention": not returned if not relevant else None,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def average(field: str) -> float | None:
        values = [row[field] for row in rows if row[field] is not None]
        return round(statistics.mean(values), 6) if values else None

    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "queries": len(rows),
        "answerable_queries": sum(bool(row["expected"]) for row in rows),
        "no_answer_queries": sum(not row["expected"] for row in rows),
        "macro_precision_returned": average("precision"),
        "macro_precision_at_k": average("precision_at_k"),
        "macro_recall_at_k": average("recall"),
        "mrr_at_k": average("reciprocal_rank"),
        "exact_set_accuracy": average("exact_set_match"),
        "no_answer_accuracy": average("correct_abstention"),
        "forbidden_hit_count": sum(len(row["forbidden_returned"]) for row in rows),
        "queries_with_forbidden_hits": sum(bool(row["forbidden_returned"]) for row in rows),
        "mean_latency_ms": average("latency_ms"),
        "p95_latency_ms": round(latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)], 6),
        "mean_estimated_context_tokens": average("estimated_context_tokens"),
        "max_estimated_context_tokens": max(row["estimated_context_tokens"] for row in rows),
    }


def evaluate(data: dict[str, Any], *, limit: int = 3, budget: int = 1200, repeats: int = 3,
             embedding_provider: EmbeddingProvider | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="coding-agent-memory-evaluation-") as directory:
        store = MemoryStore(Path(directory) / "evaluation.sqlite3")
        named = seed_store(store, data)
        id_to_key = {record.id: key for key, record in named.items()}
        live = [record for record in store.list_all()]

        def no_memory(query: str, scope: dict[str, Any]) -> list[MemoryRecord]:
            return []

        def recent_proxy(query: str, scope: dict[str, Any]) -> list[MemoryRecord]:
            # Deliberately query-independent: analogous to sending the most
            # recent context, while applying the same safe visibility policy.
            return bound_context(
                sorted((r for r in live if visible_and_active(r, scope)), key=lambda r: r.id, reverse=True),
                limit, budget,
            )

        def fts(query: str, scope: dict[str, Any]) -> list[MemoryRecord]:
            hits = store.retrieve(query, limit=limit, max_tokens=budget, **scope)
            return bound_context([hit.record for hit in hits], limit, budget)

        methods: dict[str, Callable[[str, dict[str, Any]], list[MemoryRecord]]] = {
            "no_memory": no_memory,
            "recent_records_proxy": recent_proxy,
            "scoped_fts5": fts,
        }
        tracked = TrackedEmbeddingProvider(embedding_provider) if embedding_provider else None
        if tracked is not None:
            hybrid_store = MemoryStore(store.db_path, embedding_provider=tracked)

            def hybrid(query: str, scope: dict[str, Any]) -> list[MemoryRecord]:
                hits = hybrid_store.retrieve(query, limit=limit, max_tokens=budget, **scope)
                return bound_context([hit.record for hit in hits], limit, budget)

            methods["scoped_hybrid"] = hybrid
        results = {}
        for name, retrieve in methods.items():
            rows = []
            for case in data["cases"]:
                scope = {**data["default_scope"], **case.get("scope", {})}
                failures_before = sum(tracked.failures.values()) if tracked else 0
                # One untimed warm-up. Query/document caches normally absorb
                # provider work; cache misses/failures can still perform IO.
                retrieve(case["query"], scope)
                calls_before_timing = tracked.calls if tracked else 0
                timings = []
                for _ in range(repeats):
                    started = perf_counter()
                    selected = retrieve(case["query"], scope)
                    timings.append((perf_counter() - started) * 1000)
                returned = [id_to_key[record.id] for record in selected]
                forbidden = set(case.get("forbidden", [])) | {
                    key for key, record in named.items()
                    if not visible_and_active(record, scope)
                }
                context = "\n".join(render_record(record) for record in selected)
                rows.append({
                    "id": case["id"], "difficulty": case["difficulty"],
                    "category": case["category"], "query": case["query"],
                    "scope": scope, "expected": case["expected"], "returned": returned,
                    **query_metrics(returned, case["expected"], limit),
                    "forbidden_returned": sorted(set(returned) & forbidden),
                    "latency_ms": round(statistics.median(timings), 6),
                    "estimated_context_tokens": estimated_tokens(context),
                    "embedding_failures": (sum(tracked.failures.values()) - failures_before) if tracked else 0,
                    "timed_embedding_requests": (tracked.calls - calls_before_timing) if tracked else 0,
                })
            results[name] = {
                "summary": aggregate(rows),
                "by_difficulty": {
                    level: aggregate([row for row in rows if row["difficulty"] == level])
                    for level in sorted({row["difficulty"] for row in rows})
                },
                "queries": rows,
            }
            if name == "scoped_hybrid" and tracked is not None:
                results[name]["embedding_provider"] = {
                    "fingerprint": tracked.fingerprint, "calls": tracked.calls,
                    "input_texts": tracked.input_texts, "failures_by_type": dict(tracked.failures),
                    "degraded_to_keyword_on_error": bool(tracked.failures),
                    "timed_embedding_requests": sum(row["timed_embedding_requests"] for row in rows),
                    "query_cache": "bounded process-local LRU; each query has an untimed warm-up",
                    "cold_document_embedding_cost_measured": False,
                    "billed_usage_and_cost_measured": False,
                }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": "tests/fixtures/memory_retrieval.json",
        "fixture_title": data["title"],
        "fixture_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                        "platform": platform.system(), "architecture": platform.machine()},
        "configuration": {"top_k": limit, "estimated_token_budget": budget,
                          "timing_repeats": repeats, "timing": "median of warm runs per query",
                          "token_estimator": "ceil(UTF-8 bytes / 3); heuristic, not billed model tokens"},
        "dataset": {"memories": len(data["memories"]), "queries": len(data["cases"]),
                    "difficulties": dict(Counter(c["difficulty"] for c in data["cases"])),
                    "synthetic": True, "held_out": False, "end_to_end": False},
        "limitations": [
            "Development retrieval benchmark; not unseen or end-to-end coding-task evaluation.",
            "Recent-record baseline is a safe scope-filtered context-selection proxy, not real raw conversation history.",
            "Precision, recall and MRR use answerable queries only; no-answer accuracy is reported separately.",
            "No LLM answer quality, task success, iteration counts, recovery rate, or paid token costs are measured.",
            "Latency excludes startup, embedding model loading and LLM calls; small synthetic corpus stays warm.",
            ("Hybrid retrieval explicitly requested; inspect provider failure counts before interpreting semantic results."
             if embedding_provider else "No semantic embedding provider was evaluated in this offline run."),
        ],
        "methods": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--embedding-model", help="Opt in to networked OpenRouter hybrid evaluation; requires OPENROUTER_API_KEY")
    args = parser.parse_args()
    if min(args.top_k, args.max_tokens, args.repeats) < 1:
        parser.error("top-k, max-tokens and repeats must be positive")
    provider = None
    if args.embedding_model:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            parser.error("--embedding-model requires OPENROUTER_API_KEY in the environment")
        provider = OpenRouterEmbeddingProvider(api_key=api_key, model=args.embedding_model)
    result = evaluate(load_fixture(args.fixture), limit=args.top_k, budget=args.max_tokens,
                      repeats=args.repeats, embedding_provider=provider)
    result["fixture"] = str(args.fixture)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: value["summary"] for name, value in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()

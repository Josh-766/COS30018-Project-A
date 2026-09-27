"""Token accounting and protocol-preserving compaction of working history."""
from __future__ import annotations

import json
from typing import Any


class ContextBudgetExceeded(ValueError):
    """The goal or latest/pending exchange cannot fit without losing meaning."""


def estimate_tokens(value: Any) -> int:
    """Deterministic UTF-8 heuristic, not a provider's billed token count."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, (len(text.encode('utf-8')) + 2) // 3)


def _summary(messages: list[dict[str, Any]]) -> str:
    """Extract observable outcomes, never infer a successful fix or hidden thought."""
    parts = []
    calls = {call['id']: call for m in messages for call in m.get('tool_calls', [])}
    for message in messages:
        if message['role'] == 'user':
            parts.append('User requested: ' + str(message.get('content') or '')[:200])
        elif message['role'] == 'assistant' and not message.get('tool_calls'):
            parts.append('Assistant reported (unverified): ' + str(message.get('content') or '')[:160])
        elif message['role'] == 'tool':
            call = calls.get(message.get('tool_call_id'), {})
            function = call.get('function', {})
            try:
                args = json.loads(function.get('arguments', '{}'))
                result = json.loads(message.get('content') or '{}')
            except (ValueError, TypeError):
                args, result = {}, {}
            if not isinstance(args, dict):
                args = {}
            if not isinstance(result, dict):
                result = {}
            target = str(args.get('path') or args.get('command') or args.get('query') or '')[:160]
            details = {key: result[key] for key in ('exit_code', 'error', 'written', 'saved_memory_id', 'truncated') if key in result}
            excerpt = str(result.get('stderr') or result.get('stdout') or '')[-120:]
            if excerpt:
                details['output_tail'] = excerpt
            parts.append(f"Tool {function.get('name', 'unknown')} {target}: {json.dumps(details, ensure_ascii=False)}")
    return ' | '.join(parts)[:900]


def _units(messages: list[dict[str, Any]]) -> list[tuple[list[dict[str, Any]], bool]]:
    """Split a turn into messages or complete assistant/parallel-tool exchanges."""
    units = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        calls = message.get('tool_calls', []) if message['role'] == 'assistant' else []
        if calls:
            expected = {call['id'] for call in calls}
            seen = set()
            while index < len(messages) and messages[index]['role'] == 'tool':
                group.append(messages[index])
                seen.add(messages[index].get('tool_call_id'))
                index += 1
            units.append((group, expected == seen))
        else:
            # Orphan results are never silently removed to make history look valid.
            units.append((group, message['role'] != 'tool'))
    return units


def compact_history(context: list[dict[str, Any]], summary: list[str], *,
                    history_tokens: int, summary_tokens: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the current user message and latest/pending tool exchange intact.

    Older turns are evicted first. Within the current turn, completed exchanges
    can become extractive summaries; the latest exchange remains live evidence.
    """
    turns: list[list[dict[str, Any]]] = []
    for message in context:
        if not turns or message['role'] == 'user':
            turns.append([])
        turns[-1].append(message)
    summaries = list(summary)
    while len(turns) > 1 and estimate_tokens([m for turn in turns for m in turn]) > history_tokens:
        old = turns[0]
        if not all(complete for _, complete in _units(old)):
            break
        turns.pop(0)
        summaries.append(_summary(old))
    clean = [m for turn in turns for m in turn]
    if len(turns) == 1 and estimate_tokens(clean) > history_tokens:
        current = turns[0]
        anchor = current[:1] if current and current[0]['role'] == 'user' else []
        units = _units(current[len(anchor):])
        while len(units) > 1 and estimate_tokens(anchor + [m for group, _ in units for m in group]) > history_tokens:
            removable = next((i for i, (_, complete) in enumerate(units[:-1]) if complete), None)
            if removable is None:
                break
            group, _ = units.pop(removable)
            summaries.append(_summary(group))
        clean = anchor + [m for group, _ in units for m in group]
    summaries = [item for item in summaries if item]
    while summaries and estimate_tokens(summaries) > summary_tokens:
        if len(summaries) > 1:
            summaries.pop(0)
        else:
            # Keep a bounded observation even when one summary exceeds its quota.
            summaries[0] = summaries[0][:max(0, len(summaries[0]) // 2)]
            if not summaries[0]:
                summaries.clear()
    if estimate_tokens(clean) > history_tokens:
        raise ContextBudgetExceeded(
            'The current request or latest/pending tool exchange exceeds the history budget. '
            'Use smaller file pages or edits, or increase MEMORY_HISTORY_TOKENS.'
        )
    return clean, summaries

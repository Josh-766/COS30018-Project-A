"""Bound model-visible tool observations without breaking their JSON structure."""

import json
from dataclasses import dataclass


_STATUS_FIELDS = {'status', 'exit_code', 'timed_out', 'error', 'errors', 'stderr', 'truncated'}


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


@dataclass
class _JSONText:
    """A tool field which itself contains serialized JSON; keep its string type."""

    value: object


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, (dict, list)):
            return _JSONText(_expand(parsed))
    return value


def _materialize(value):
    if isinstance(value, _JSONText):
        return _dump(_materialize(value.value))
    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    return value


def _excerpt(value, length):
    if len(value) <= length:
        return value
    marker = '\n...[omitted]...\n'
    available = max(0, length - len(marker))
    head = (available + 1) // 2
    tail = available - head
    return value[:head] + marker + (value[-tail:] if tail else '')


def _candidates(value):
    """Yield mutable containers and fields, leaving status metadata intact."""
    if isinstance(value, _JSONText):
        yield from _candidates(value.value)
    elif isinstance(value, (dict, list)):
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in entries:
            if isinstance(item, str) and len(item) > 48 and key != 'exit_code':
                yield len(item), value, key, item
            elif isinstance(item, list) and len(item) > 1:
                # Trim lists by complete entries, never by slicing encoded JSON.
                yield len(_dump(_materialize(item))), value, key, item
            elif isinstance(item, dict) and len(set(item) - _STATUS_FIELDS) > 1:
                yield len(_dump(_materialize(item))), value, key, item
            yield from _candidates(item)


def _subset(value, length):
    if isinstance(value, str):
        return _excerpt(value, length)
    if isinstance(value, list):
        return value[:length]
    keys = [key for key in value if key not in _STATUS_FIELDS][:length]
    return {key: item for key, item in value.items() if key in _STATUS_FIELDS or key in keys}


def bounded_tool_output(output: str, limit: int = 4000) -> str:
    """Keep status and head/tail diagnostics in a valid, size-bounded observation.

    Already-small observations are unchanged. Formatting compaction alone does
    not mark an observation truncated. Structured fields and serialized JSON
    fields retain valid structure; oversized lists lose complete trailing items.
    The minimum budget leaves room for useful status and truncation metadata.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 256:
        raise ValueError('Tool output limit must be an integer of at least 256 characters')
    if len(output) <= limit:
        return output
    try:
        parsed = json.loads(output)
    except ValueError:
        parsed = {'excerpt': output}
    if not isinstance(parsed, dict):
        parsed = {'result': parsed}
    value = _expand(parsed)

    def encoded():
        return _dump(_materialize(value))

    if len(encoded()) <= limit:
        return encoded()
    value['truncated'] = True
    while len(encoded()) > limit:
        candidates = list(_candidates(value))
        if not candidates:
            # Pathological dictionaries can exceed the limit just in key names.
            # Keep diagnostic/status fields, dropping unrelated payload fields.
            removable = [key for key in value if key not in _STATUS_FIELDS]
            if not removable:
                raise ValueError('Tool status metadata exceeds the output budget')
            key = max(removable, key=lambda item: len(_dump(_materialize(value[item]))) + len(item))
            del value[key]
            continue
        _, parent, key, original = max(candidates, key=lambda item: item[0])
        maximum = len(set(original) - _STATUS_FIELDS) if isinstance(original, dict) else len(original)
        minimum = 48 if isinstance(original, str) else 1
        # Find the largest whole field/list prefix that fits the final encoding,
        # accounting for quotes, escaping and nested serialized JSON overhead.
        low, high = minimum, maximum - 1
        best = minimum
        while low <= high:
            mid = (low + high) // 2
            parent[key] = _subset(original, mid)
            if len(encoded()) <= limit:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        parent[key] = _subset(original, best)
    return encoded()

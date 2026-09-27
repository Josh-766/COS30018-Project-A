"""One serializer for budgeting and injecting untrusted historical memory."""

from __future__ import annotations

import json
from typing import Any


def encode_memory_envelope(records: list, working_context: Any = None) -> str:
    """Encode the exact user-message content used by the model request.

    Budget this output rather than the inner memory object: JSON quoting and
    escaped markup can expand strings substantially. JSON round trips preserve
    all original data; embedded markup never becomes an envelope delimiter.
    """
    envelope: dict[str, Any] = {
        "type": "historical_memory_context",
        "records": records,
    }
    if working_context is not None:
        envelope["working_context"] = working_context
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

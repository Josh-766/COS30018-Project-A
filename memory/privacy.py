"""Best-effort credential filtering that preserves code and JSON structure.

This is a persistence safeguard, not a secret scanner or a guarantee that arbitrary
secrets are detectable. Recognizable credentials and literal credential assignments
are removed; type annotations, calls and environment references are source code.
Ambiguous bare assignments (``api_key=short``) are treated as configuration secrets.
Use environment references for credentials and never rely on this filter alone.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
_KEY = (
    r"(?:[a-zA-Z_][a-zA-Z0-9_]*_)?"
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|private[_ -]?key|"
    r"secret[_ -]?key|password|secret|authorization)"
)
_SENSITIVE_KEY = re.compile(rf"{_KEY}\Z", re.IGNORECASE)
_QUOTED = r'''(?i:(?:fr|rf|br|rb|r|b|u|f)?)(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\[^\r\n]|[^"\\\r\n])*"|'(?:\\[^\r\n]|[^'\\\r\n])*')'''
_ASSIGNMENT = re.compile(
    rf"(?P<prefix>\b{_KEY}\b[\"']?(?:\s*\])?\s*(?P<operator>:(?!=)|=(?!=))\s*)"
    rf"(?P<value>{_QUOTED}|[^\s,;\"'{{}}<>]+)",
    re.IGNORECASE,
)
# The first expression is the annotation, not a credential. This separately finds
# literal defaults in ``password: str = '...'`` (including function parameters).
_ANNOTATED_ASSIGNMENT = re.compile(
    rf"(?P<prefix>\b{_KEY}\s*:\s*[A-Za-z_\"'][\w.\[\], |'\"]*?\s*=(?!=)\s*)"
    rf"(?P<value>{_QUOTED}|[^\s,;\"'{{}}<>]+)",
    re.IGNORECASE,
)
_PEM = re.compile(
    r"-----BEGIN (?P<key_type>(?:[A-Z ]+ )?PRIVATE KEY)-----"
    r"[\s\S]*?(?:-----END (?P=key_type)-----|$)", re.IGNORECASE,
)
_TOKEN = re.compile(
    r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|"
    r"github_pat_[A-Za-z0-9_]{12,}|AKIA[A-Z0-9]{16})\b"
)
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@[^\s\"'<>]+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+(?!\[REDACTED\])[^\s\"',;}]+", re.IGNORECASE)
_AUTH_HEADER = re.compile(r"(?im)^(?P<prefix>Authorization:\s*)(?P<value>[^\r\n]+)")
_REFERENCE = re.compile(
    r"(?:\$\{[^{}\n]+\}|\$[A-Za-z_][\w]*|\{\{[^{}\n]+\}\})\Z"
)
_TYPE = re.compile(
    r"(?:str|int|float|bool|bytes|object|Any|None|String|String\?|string|number|boolean|"
    r"Optional\[.*|Union\[.*|Annotated\[.*|SecretStr|typing\.[\w\[\], .|]+)\Z"
)


def _source_spans(value: str, *, source: bool | None = None):
    """Find source boundaries, including incomplete or indented file pages."""
    try:
        tree = ast.parse(value)
    except (SyntaxError, ValueError, RecursionError):
        tree = None
    lines = value.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def position(line: int, byte_column: int) -> int:
        # AST columns count UTF-8 bytes; regex offsets count Unicode codepoints.
        return offsets[line - 1] + len(lines[line - 1].encode('utf-8')[:byte_column].decode('utf-8'))

    annotations, literals, references = [], [], []
    nodes = list(ast.walk(tree)) if tree is not None else []
    is_code = source is True or (source is None and (
        any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom, ast.Call)) for node in nodes)
        or bool(re.search(r'(?m)^\s*(?:async\s+)?(?:def|class)\s+\w+', value))))
    for node in nodes:
        annotation = getattr(node, 'annotation', None)
        if source is not False and annotation is not None and hasattr(annotation, 'end_lineno'):
            left = position(annotation.lineno, annotation.col_offset)
            right = position(annotation.end_lineno, annotation.end_col_offset)
            if is_code or _TYPE.fullmatch(value[left:right].strip('\"\'')):
                annotations.append((left, right))
        if isinstance(node, (ast.Constant, ast.JoinedStr)) and isinstance(getattr(node, 'value', ''), str):
            literals.append((position(node.lineno, node.col_offset),
                             position(node.end_lineno, node.end_col_offset)))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            expression = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            clear_reference = is_code or any(isinstance(target, (ast.Attribute, ast.Subscript))
                or (isinstance(target, ast.Name) and isinstance(expression, ast.Name)
                    and target.id == expression.id) for target in targets)
            if clear_reference and isinstance(expression, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                references.append((position(expression.lineno, expression.col_offset),
                                   position(expression.end_lineno, expression.end_col_offset)))
    # tokenize understands indentation and already-complete string tokens even
    # when a page ends halfway through a function or parenthesized expression.
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(value).readline):
            tokens.append(token)
            if token.type == tokenize.STRING:
                literals.append((offsets[token.start[0] - 1] + token.start[1],
                                 offsets[token.end[0] - 1] + token.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    significant = [token for token in tokens if token.type not in {
        tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
        tokenize.ENCODING, tokenize.ENDMARKER}]
    for index, token in enumerate(significant[:-2]):
        # An unquoted identifier followed by ':' is an annotation. Quoted dict
        # keys ("password": "secret") must still be treated as credentials.
        if (token.type == tokenize.NAME and _SENSITIVE_KEY.fullmatch(token.string)
                and significant[index + 1].string == ':' and tree is None and is_code):
            annotation = significant[index + 2]
            if annotation.type in {tokenize.NAME, tokenize.STRING}:
                annotations.append((offsets[annotation.start[0] - 1] + annotation.start[1],
                                    offsets[annotation.end[0] - 1] + annotation.end[1]))
        if (is_code and token.type == tokenize.NAME and _SENSITIVE_KEY.fullmatch(token.string)
                and significant[index + 1].string == '='):
            expression = significant[index + 2]
            if expression.type == tokenize.NAME:
                references.append((offsets[expression.start[0] - 1] + expression.start[1],
                                   offsets[expression.end[0] - 1] + expression.end[1]))
    return annotations, literals, references


def _mask(value: str, preserve_layout: bool) -> str:
    """Mask source without changing the original line/column coordinates."""
    if not preserve_layout:
        return REDACTED
    return re.sub(r'[^\r\n]+', lambda match: (REDACTED + ' ' * len(match.group(0)))[:len(match.group(0))]
                  if len(match.group(0)) >= len(REDACTED) else '*' * len(match.group(0)), value)


def _is_reference(value: str) -> bool:
    return bool(_REFERENCE.fullmatch(value.strip()))


def _crosses_literal_boundary(match: re.Match[str], literals: Sequence[tuple[int, int]]) -> bool:
    operator_position = (match.start('operator') if 'operator' in match.re.groupindex
                         else match.start())
    return any(left <= operator_position < right and match.end() >= right
               for left, right in literals)


def _replace_assignment(match: re.Match[str], annotations: list[tuple[int, int]],
                        literals: Sequence[tuple[int, int]] = (), *,
                        preserve_layout: bool = False, allow_types: bool = True) -> str:
    value = match.group('value')
    start = match.start('value')
    if any(left <= start < right for left, right in annotations):
        return match.group(0)
    # A prompt label such as input("Enter password: ") is inside a string;
    # its closing quote must never become an opening credential quote. Still
    # redact a complete password=... assignment contained inside a log string.
    if _crosses_literal_boundary(match, literals):
        return match.group(0)
    literal = re.match(r"(?i)(fr|rf|br|rb|r|b|u|f)?([\"'])", value)
    literal_prefix = (literal.group(1) or '') if literal else ''
    quoted = value[len(literal_prefix):]
    quote = '"""' if quoted.startswith('"""') else "'''" if quoted.startswith("'''") else quoted[0] if quoted[0] in "\"'" else ''
    raw = quoted[len(quote):-len(quote)] if quote else value
    if raw == REDACTED or not raw or _is_reference(raw):
        return match.group(0)
    if 'f' in literal_prefix.casefold() and ('{' in raw or '}' in raw):
        return match.group(0)
    if not quote:
        # Expressions are code: do not turn input(...), os.environ[...], JS
        # process.env.KEY, parameter defaults, etc. into invalid source text.
        if any(char in value for char in '([.$') or value in {'None', 'null', 'True', 'False', 'true', 'false'}:
            return match.group(0)
        operator = match.groupdict().get('operator')
        if allow_types and operator == ':' and _TYPE.fullmatch(value.rstrip('):,')):
            return match.group(0)
    return match.group('prefix') + literal_prefix + quote + _mask(raw, preserve_layout) + quote


def _redact_assignments(pattern: re.Pattern[str], value: str, *, source: bool | None = None,
                        preserve_layout: bool = False) -> str:
    annotations, literals, references = _source_spans(value, source=source)
    result, copied, position = [], 0, 0
    while match := pattern.search(value, position):
        if _crosses_literal_boundary(match, literals):
            # Resume after the misleading label, not after its overlong match:
            # an actual secret assignment may follow on the same source line.
            position = match.start() + 1
            continue
        result.extend((value[copied:match.start()], _replace_assignment(
            match, annotations + references, literals, preserve_layout=preserve_layout,
            allow_types=source is not False)))
        copied = position = match.end()
    result.append(value[copied:])
    return ''.join(result)


def _redact_plain_text(value: str, *, source: bool | None = None, preserve_layout: bool = False) -> str:
    # Remove multiline material before any truncation at the call site. Preserve
    # surrounding literal quotes so JSON and source stay parseable.
    replace = lambda match: _mask(match.group(0), preserve_layout)
    value = _PEM.sub(replace, value)
    value = _URL.sub(replace, value)
    value = _TOKEN.sub(replace, value)
    value = _BEARER.sub(replace, value)
    value = _AUTH_HEADER.sub(
        lambda match: match.group(0) if _TYPE.fullmatch(match.group('value').strip('\"\''))
        else _replace_assignment(match, [], preserve_layout=preserve_layout),
        value,
    )
    if not _ASSIGNMENT.search(value):
        return value
    value = _redact_assignments(_ASSIGNMENT, value, source=source, preserve_layout=preserve_layout)
    if not _ANNOTATED_ASSIGNMENT.search(value):
        return value
    # Recompute positions after replacements shifted the source text.
    return _redact_assignments(_ANNOTATED_ASSIGNMENT, value, source=source, preserve_layout=preserve_layout)


def _schema(value: Mapping[Any, Any]) -> bool:
    # A property named password commonly contains a JSON Schema definition.
    # A typed credential wrapper ({"type": "string", "value": "secret"}) is
    # still a credential object; an arbitrary 'type' key cannot exempt it.
    schema_keys = {'type', '$ref', '$id', '$schema', '$anchor', '$defs', 'definitions',
        'properties', 'patternProperties', 'additionalProperties', 'unevaluatedProperties',
        'required', 'dependentRequired', 'dependentSchemas', 'propertyNames',
        'anyOf', 'oneOf', 'allOf', 'not', 'if', 'then', 'else',
        'title', 'description', 'default', 'const', 'examples', 'enum', 'format',
        'readOnly', 'writeOnly', 'deprecated', 'minLength', 'maxLength', 'pattern',
        'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
        'items', 'prefixItems', 'contains', 'minItems', 'maxItems', 'uniqueItems',
        'minContains', 'maxContains', 'minProperties', 'maxProperties', '$comment',
        'contentEncoding', 'contentMediaType', 'contentSchema'}
    if not set(value) <= schema_keys:
        return False
    if any(key in value for key in ('$ref', 'properties', 'anyOf', 'oneOf', 'allOf')):
        return True
    kind = value.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    return bool(kinds) and all(isinstance(item, str) and item in {
        'null', 'boolean', 'object', 'array', 'number', 'integer', 'string'} for item in kinds)


def redact_source_text(value: str, *, preserve_layout: bool = False) -> str:
    """Filter a known source file; optionally retain original page coordinates.

    Apply to the complete bounded file before pagination. Filtering isolated
    fragments cannot recognize a credential whose label is on an earlier page.
    """
    return _redact_plain_text(value, source=True, preserve_layout=preserve_layout)


def is_source_path(path: Any) -> bool:
    return isinstance(path, str) and path.rsplit('.', 1)[-1].casefold() in {
        'py', 'pyi', 'js', 'jsx', 'ts', 'tsx', 'java', 'c', 'h', 'cpp', 'hpp',
        'go', 'rs', 'rb', 'php', 'swift', 'cs', 'kt', 'kts'}


def _file_text(value: str, path: str) -> str:
    if is_source_path(path):
        return redact_source_text(value)
    # JSON configuration needs recursive credential-object handling. YAML and
    # .env files must not mistake a password key for a Python annotation.
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        return redact_sensitive_text(value)
    return _redact_plain_text(value, source=False)


def redact_file_layout(value: str, path: str) -> str:
    """Apply shared JSON/source rules before paging, with stable coordinates."""
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        safe = sanitize_json_value(decoded)
        if safe != decoded:
            decoder = json.JSONDecoder()
            replacements = []

            def skip_space(position):
                while position < len(value) and value[position].isspace():
                    position += 1
                return position

            def scalar(position, original, sanitized):
                _, end = decoder.raw_decode(value, position)
                if sanitized != original:
                    if isinstance(original, str):
                        replacement = '"' + _mask(value[position + 1:end - 1], True) + '"'
                    else:
                        # Keep numeric/boolean credential fields valid JSON
                        # without retaining their values or shifting cursors.
                        width = end - position
                        replacement = ('null' if width >= 4 else '0').ljust(width)
                    replacements.append((position, end, replacement))
                return end

            def walk(position, original, sanitized):
                position = skip_space(position)
                if isinstance(original, dict):
                    position = skip_space(position + 1)
                    for key, item in original.items():
                        safe_key = redact_sensitive_text(key)
                        position = skip_space(scalar(position, key, safe_key))
                        position = walk(position + 1, item, sanitized[safe_key])  # ':'
                        position = skip_space(position)
                        if value[position] == ',':
                            position = skip_space(position + 1)
                    return position + 1  # '}'
                if isinstance(original, list):
                    position = skip_space(position + 1)
                    for item, safe_item in zip(original, sanitized):
                        position = skip_space(walk(position, item, safe_item))
                        if value[position] == ',':
                            position = skip_space(position + 1)
                    return position + 1  # ']'
                return scalar(position, original, sanitized)

            walk(0, decoded, safe)
            pieces, copied = [], 0
            for start, end, replacement in replacements:
                pieces.extend((value[copied:start], replacement))
                copied = end
            pieces.append(value[copied:])
            value = ''.join(pieces)
    return _redact_plain_text(value, source=is_source_path(path), preserve_layout=True)


def _sanitize_credential_field(value: Any) -> Any:
    if isinstance(value, Mapping):
        if _schema(value):
            return {
                redact_sensitive_text(str(key)): (
                    _sanitize_credential_field(item)
                    if key in {'default', 'const', 'examples', 'enum'}
                    else sanitize_json_value(item)
                )
                for key, item in value.items()
            }
        return {redact_sensitive_text(str(key)): _sanitize_credential_field(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_credential_field(item) for item in value]
    if value is None or value == '' or value == REDACTED:
        return value
    if isinstance(value, str) and _is_reference(value):
        return value
    return REDACTED


def sanitize_json_value(value: Any) -> Any:
    """Return credential-safe nested data, preserving schema and code leaves."""
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            redact_sensitive_text(str(key)): (
                _sanitize_credential_field(item) if _SENSITIVE_KEY.fullmatch(str(key))
                else _file_text(item, value['path']) if key == 'content' and isinstance(item, str)
                    and isinstance(value.get('path'), str)
                else sanitize_json_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


def redact_sensitive_text(value: str) -> str:
    """Redact actual credential values without changing ordinary source code.

    JSON containers are decoded first so escaped code and nested metadata use
    exactly the same rules. Safe input retains its original text formatting.
    """
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        sanitized = sanitize_json_value(decoded)
        return value if sanitized == decoded else json.dumps(sanitized, ensure_ascii=False)
    return _redact_plain_text(value)


def contains_sensitive_value(value: Any) -> bool:
    """Use the redaction detector for validation; no divergent regex allowlist."""
    if isinstance(value, str):
        return redact_sensitive_text(value) != value
    if isinstance(value, Mapping):
        return any(
            contains_sensitive_value(str(key)) or
            (_sanitize_credential_field(item) != item if _SENSITIVE_KEY.fullmatch(str(key))
             else _file_text(item, value['path']) != item if key == 'content' and isinstance(item, str)
                 and isinstance(value.get('path'), str)
             else contains_sensitive_value(item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_sensitive_value(item) for item in value)
    return False

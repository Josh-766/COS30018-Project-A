"""Register tool definitions and dispatch model tool calls."""
import json
from skill_library import SkillLibrary
from memory import forget_memory, save_memory, search_memories
from .terminal_use import TOOL as COMMAND_TOOL, execute_command
from .read_file import TOOL as READ_FILE_TOOL, read_file
from .write_file import TOOL as WRITE_FILE_TOOL, write_file
from .list_files import TOOL as LIST_FILES_TOOL, list_files

SPECS = {
    'run_command': (COMMAND_TOOL, execute_command),
    'read_file': (READ_FILE_TOOL, read_file),
    'write_file': (WRITE_FILE_TOOL, write_file),
    'list_files': (LIST_FILES_TOOL, list_files),
}


def _memory_tool(name, description, field):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string"}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


MEMORY_SPECS = {
    "save_memory": (
        _memory_tool("save_memory", "Remember a useful, confirmed project fact or user preference across conversations. Maximum 2000 characters; never save credentials.", "content"),
        save_memory,
    ),
    "search_memories": (
        _memory_tool("search_memories", "Recall up to five saved memories for this user and project using specific keywords. Use when past decisions or preferences would help.", "query"),
        search_memories,
    ),
    "forget_memory": (
        _memory_tool("forget_memory", "Delete a saved memory by its ID when the user asks to forget it or confirms it is obsolete.", "memory_id"),
        forget_memory,
    ),
}


def _skill_tool(name, description, fields):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
                "required": fields,
                "additionalProperties": False,
            },
        },
    }


SKILL_SPECS = {
    "list_skills": _skill_tool(
        "list_skills", "List available coding skills and when to use each. Returns metadata, not full instructions.", []),
    "read_skill": _skill_tool(
        "read_skill", "Read a skill's full SKILL.md instructions before applying it. Use a name from the catalog; also lists supporting resources.", ["name"]),
    "read_skill_resource": _skill_tool(
        "read_skill_resource", "Read a UTF-8 supporting file relative to a skill folder, such as scripts/with_server.py. This only reads; it never executes scripts.", ["name", "path"]),
}


def list_tools():
    return [definition for definition, execute in (SPECS | MEMORY_SPECS).values()] + list(SKILL_SPECS.values())


def execute_tool(tool_call, *, sandbox, store=None, memory_context=None, skills=None):
    try:
        function = tool_call['function']
        name = function['name']
        specs = SPECS | MEMORY_SPECS
        definitions = {key: value[0] for key, value in specs.items()} | SKILL_SPECS
        if name not in definitions:
            raise ValueError('execute_tool: unknown tool')
        arguments = json.loads(function['arguments'])
        fields = definitions[name]['function']['parameters']['required']
        if (not isinstance(arguments, dict) or set(arguments) != set(fields)
                or any(not isinstance(v, str) for v in arguments.values())):
            raise ValueError('execute_tool: invalid arguments')
        if name in SKILL_SPECS:
            library = skills if skills is not None else SkillLibrary()
            result = getattr(library, name)(**arguments)
        elif name in MEMORY_SPECS:
            if store is None or memory_context is None:
                raise ValueError('memory tools require a graph store and memory context')
            result = specs[name][1](store, memory_context, **arguments)
        else:
            result = specs[name][1](arguments, sandbox)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({'error': f'execute_tool: {exc}'})

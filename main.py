"""Interactive coding agent with scoped persistent memory and bounded tool runs."""
import argparse
import getpass
import json
import os
from pathlib import Path

from agent_roles import send_to_coder
from memory import MemoryStore
from memory.observations import bounded_tool_output
from memory.service import ContextBudgetExceeded, MemoryService, project_identity, redact
from scripts.tools.tool_registry import execute_tool
from scripts.tools.sandbox_setup import create_session


MEMORY_HELP = '''Memory commands:
  /remember <text>             save a project decision
  /remember:<type> <text>      save a requirement/preference/constraint/decision
  /memories                   inspect visible records and their status
  /search <query>              inspect ranked recall and provenance
  /correct <id> <text>         supersede a memory with a correction
  /forget <id>                 delete a durable record
  /sessions | /new | /resume <id>
  /task <goal>                start a new task in this conversation
  /continue [message]         continue the current task, including an answered one
  /delete-session <id>         erase that session's transcript/checkpoint/log
  /state | /trace              inspect current task and recent memory events
  /help                       show this list
'''


def memory_command(service: MemoryService, text: str) -> bool:
    if not text.startswith('/'):
        return False
    command, _, argument = text.partition(' ')
    try:
        if command == '/help':
            print(MEMORY_HELP)
        elif command == '/remember' or command.startswith('/remember:'):
            kind = command.split(':', 1)[1] if ':' in command else 'decision'
            record = service.remember(argument, kind)
            print(f'Memory: saved {record.memory_type} #{record.id}')
        elif command == '/memories':
            records = service.visible_records()
            print('\n'.join(f'#{r.id} [{r.memory_type}/{r.status}] {r.content}' for r in records) or 'No visible memories.')
        elif command == '/search':
            print(json.dumps(json.loads(service.build_context(argument)), indent=2, ensure_ascii=False))
        elif command == '/forget':
            service.forget(int(argument))
            print('Memory deleted. Session transcripts are separate; /delete-session removes those.')
        elif command == '/correct':
            identifier, content = argument.split(' ', 1)
            record = service.correct(int(identifier), content)
            print(f'Memory: saved correction #{record.id}')
        elif command == '/sessions':
            print(json.dumps(service.sessions(), indent=2))
        elif command in {'/new', '/resume'}:
            if command == '/resume' and not argument.strip():
                raise ValueError('Usage: /resume <session-id>; inspect /sessions first')
            service.switch_session(argument.strip() if command == '/resume' else None)
            print(f'Session: {service.session.session_id}')
            if command == '/resume':
                print('Restored conversation and task state. Sandbox files are not restored; inspect before continuing.')
        elif command == '/delete-session':
            print('Session deleted.' if service.delete_session(argument.strip()) else 'Session not found.')
        elif command == '/state':
            from dataclasses import asdict
            print(json.dumps(asdict(service.session.task) if service.session.task else {}, indent=2))
        elif command == '/trace':
            print(json.dumps(service.events(), indent=2))
        else:
            print('Unknown command. Use /help.')
    except (ValueError, KeyError, RuntimeError) as error:
        print(f'Memory: {error}')
    return True


def run_turn(service: MemoryService, text: str, *, sandbox_factory, max_rounds: int = 12,
             new_task: bool = False, continue_task: bool = False) -> str:
    """Run one adaptive task; tools are never replayed from restored checkpoints."""
    if max_rounds < 1:
        raise ValueError('Tool round limit must be positive')
    service.begin_turn(text, new_task=new_task, continue_task=continue_task)
    try:
        saved = service.observe_user(text)
        if saved:
            print(f'Memory: remembered {saved.memory_type} #{saved.id}')
    except ValueError:
        print('Memory: durable write skipped because the request contains sensitive or invalid content.')
    for iteration in range(max_rounds):
        result = send_to_coder(None, context=service.session.context,
            memory_context=service.build_context(service.recall_query(text)),
            environment_notice=service.environment_notice, extra_tools=service.tool_definitions())
        service.checkpoint(result['context'])
        service.event('model_response', {'iteration': iteration + 1, 'model': result.get('model'),
            'usage': result.get('usage'), 'finish_reason': result.get('finish_reason')})
        if not result['has_tool_call']:
            status = 'answered' if result.get('text') and result.get('finish_reason') != 'length' else 'incomplete'
            service.finish_task(status)
            return result.get('text') or 'The model returned no final answer.'
        for call in result['tool_calls']:
            name = call['function']['name']
            print(f'Running tool: {name}')
            if name in {tool['function']['name'] for tool in service.tool_definitions()}:
                # memory_search already fits the exact memory-envelope budget.
                output = redact(service.execute_tool(call))
            else:
                output = execute_tool(call, sandbox=sandbox_factory())
                service.observe_tool(call, output)
                output = bounded_tool_output(redact(output))
            print(output)
            service.checkpoint(service.session.context + [
                {'role': 'tool', 'tool_call_id': call['id'], 'content': output}])
    service.finish_task('stopped', 'Maximum tool rounds reached')
    return f'Stopped after {max_rounds} model/tool rounds. Inspect /state and /trace before continuing.'


def finish_failed_task(service: MemoryService, status: str, error: str) -> None:
    """A broken/stale checkpoint must not hide the original runtime failure."""
    try:
        service.finish_task(status, error)
    except Exception as checkpoint_error:
        print(f'Checkpoint unavailable ({type(checkpoint_error).__name__}). Use /resume {service.session.session_id} to reload, or /new.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', help='Resume a session in the current project/user scope')
    parser.add_argument('--db', help='SQLite memory database path')
    options = parser.parse_args()
    provider = None
    # Explicit opt-in only: no remote embedding calls in the default setup.
    if os.getenv('MEMORY_EMBEDDING_MODEL'):
        from memory.embeddings import OpenRouterEmbeddingProvider
        provider = OpenRouterEmbeddingProvider(api_key=os.getenv('OPENROUTER_API_KEY', ''),
                                               model=os.environ['MEMORY_EMBEDDING_MODEL'])
    store = MemoryStore(options.db, embedding_provider=provider)
    service = MemoryService(store, project_id=os.getenv('MEMORY_PROJECT_ID') or project_identity(Path.cwd()),
        user_id=os.getenv('MEMORY_USER_ID') or getpass.getuser(), session_id=options.resume,
        history_tokens=int(os.getenv('MEMORY_HISTORY_TOKENS', '6000')),
        memory_tokens=int(os.getenv('MEMORY_CONTEXT_TOKENS', '1800')))
    max_rounds = int(os.getenv('AGENT_MAX_ROUNDS', '12'))
    sandbox = None

    def get_sandbox():
        nonlocal sandbox
        if sandbox is None:
            sandbox = create_session(Path.cwd(), excluded_paths=[store.db_path])
        return sandbox

    print(f'Session: {service.session.session_id}\nProject: {service.project_id}')
    if service.environment_notice:
        print(service.environment_notice)
    print("Type 'exit' to stop; /help for memory commands. Memory inspection works without an API key.")
    try:
        while True:
            try:
                text = input('\nYou: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.casefold() in {'exit', 'quit'}:
                break
            command, _, argument = text.partition(' ')
            new_task, continue_task = command == '/task', command == '/continue'
            if new_task or continue_task:
                text = argument.strip() or ('Continue the current task.' if continue_task else '')
                if not text:
                    print('Usage: /task <goal>')
                    continue
            elif not text or memory_command(service, text):
                continue
            if not os.getenv('OPENROUTER_API_KEY'):
                print('Set OPENROUTER_API_KEY in .env to call the coder. Memory commands remain available.')
                continue
            try:
                answer = run_turn(service, text, sandbox_factory=get_sandbox, max_rounds=max_rounds,
                                  new_task=new_task, continue_task=continue_task)
                print(f'\nCoder: {answer}')
            except KeyboardInterrupt:
                finish_failed_task(service, 'interrupted', 'Interrupted by user')
                break
            except Exception as error:
                # Errors may contain request payloads; log their class, not credentials.
                finish_failed_task(service, 'failed', type(error).__name__)
                if isinstance(error, ContextBudgetExceeded):
                    print(str(error))
                else:
                    print(f'Task failed ({type(error).__name__}). Inspect /state; use /new for a fresh context.')
    finally:
        if sandbox is not None:
            sandbox.terminate()


if __name__ == '__main__':
    main()

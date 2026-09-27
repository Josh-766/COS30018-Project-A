"""Exercise repaired memory behavior through the actual offline CLI task loop."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from main import run_turn
from memory import MemoryStore
from memory.service import MemoryService, estimate_tokens


def tool_call(identifier, name, **arguments):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


def completion(context, text='Done'):
    return {'has_tool_call': False, 'tool_calls': [], 'text': text,
            'context': context + [{'role': 'assistant', 'content': text}],
            'finish_reason': 'stop'}


def request_tool(context, call):
    return {'has_tool_call': True, 'tool_calls': [call],
            'context': context + [{'role': 'assistant', 'content': None, 'tool_calls': [call]}],
            'finish_reason': 'tool_calls'}


class MemoryLoopRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.db'
        self.store = MemoryStore(self.path)
        self.service = MemoryService(self.store, project_id='pagination-project', user_id='alice')

    def assert_protocol_complete(self, context):
        pending = set()
        for message in context:
            if message['role'] == 'assistant':
                self.assertFalse(pending, 'An assistant response preceded a pending tool result')
                pending.update(call['id'] for call in message.get('tool_calls', []))
            elif message['role'] == 'tool':
                identifier = message['tool_call_id']
                self.assertIn(identifier, pending, 'Tool result has no retained matching call')
                pending.remove(identifier)
            else:
                self.assertFalse(pending, 'A user message interrupted a pending tool exchange')
        self.assertFalse(pending, 'Model input contains an unresolved tool call')

    def prepare_unfinished_task(self):
        task = self.service.start_task('Implement pagination for the SQLite API')
        self.service.update_task(current_plan=['Inspect routes', 'Implement pagination', 'Run tests'],
                                 active_step='Implement pagination', artifacts=['src/routes.py'])
        self.service.observe_tool(tool_call('initial-test', 'run_command', command='pytest tests/test_pages.py'),
                                  json.dumps({'exit_code': 1, 'stderr': 'AssertionError: page size'}))
        self.service.checkpoint()
        saved = self.store.add('Pagination page size must remain 25 for this task', 'requirement',
                               project_id=self.service.project_id, user_id=self.service.user_id,
                               session_id=self.service.session.session_id, task_id=task.task_id,
                               source_type='user', reliability=1.0, importance=0.95)
        resumed = MemoryService(MemoryStore(self.path), project_id=self.service.project_id,
                                user_id=self.service.user_id, session_id=self.service.session.session_id)
        return resumed, saved

    def test_run_turn_continues_resumed_task_and_keeps_its_scoped_memory(self):
        resumed, saved = self.prepare_unfinished_task()
        original = asdict(resumed.session.task)
        self.assertEqual(original['status'], 'interrupted')
        self.assertEqual(original['retry_count'], 1)
        inspected = []

        def coder(text, **kwargs):
            self.assertIsNone(text)
            current = asdict(resumed.session.task)
            for key in ('task_id', 'user_goal', 'current_plan', 'active_step', 'artifacts',
                        'retry_count', 'pending_failures'):
                self.assertEqual(current[key], original[key], key)
            self.assertEqual(current['status'], 'running')
            self.assertIn(saved.id, [entry['id'] for entry in json.loads(kwargs['memory_context'])['memories']])
            self.assert_protocol_complete(kwargs['context'])
            self.assertEqual(kwargs['context'][-1]['content'], 'Continue')
            inspected.append(True)
            return completion(kwargs['context'], 'Pagination continued')

        with patch('main.send_to_coder', side_effect=coder), patch('builtins.print'):
            answer = run_turn(resumed, 'Continue', sandbox_factory=lambda: self.fail('No tools should replay'))
        self.assertEqual(answer, 'Pagination continued')
        self.assertEqual(inspected, [True])
        self.assertEqual(resumed.session.task.task_id, original['task_id'])
        self.assertEqual(resumed.session.task.status, 'answered')

    def test_explicit_new_task_resets_state_and_hides_old_task_scoped_memory(self):
        resumed, saved = self.prepare_unfinished_task()
        original_id = resumed.session.task.task_id

        def coder(text, **kwargs):
            task = resumed.session.task
            self.assertNotEqual(task.task_id, original_id)
            self.assertEqual(task.user_goal, 'Build a new export command')
            self.assertEqual(task.current_plan, [])
            self.assertEqual(task.artifacts, [])
            self.assertEqual(task.retry_count, 0)
            self.assertEqual(task.pending_failures, {})
            self.assertNotIn(saved.id, [entry['id'] for entry in json.loads(kwargs['memory_context'])['memories']])
            return completion(kwargs['context'])

        with patch('main.send_to_coder', side_effect=coder), patch('builtins.print'):
            run_turn(resumed, 'Build a new export command', new_task=True,
                     sandbox_factory=lambda: self.fail('No tools requested'))

    def test_twelve_large_file_exchanges_compact_with_complete_protocol_and_summary(self):
        model_inputs = []
        tool_count = 12

        def coder(text, **kwargs):
            context = kwargs['context']
            self.assert_protocol_complete(context)
            self.assertLessEqual(estimate_tokens(context), 6000)
            model_inputs.append(deepcopy(kwargs))
            index = len(model_inputs) - 1
            if index == tool_count:
                return completion(context, 'All twelve files inspected')
            return request_tool(context, tool_call(f'read-{index}', 'read_file', path=f'src/file_{index}.py'))

        output = json.dumps({'content': '#' + 'a' * 3499, 'truncated': False})
        with patch('main.send_to_coder', side_effect=coder), \
                patch('main.execute_tool', return_value=output) as execute, patch('builtins.print'):
            answer = run_turn(self.service, 'Inspect all twelve source files before explaining the architecture',
                              sandbox_factory=lambda: object(), max_rounds=tool_count + 1)
        self.assertEqual(answer, 'All twelve files inspected')
        self.assertEqual(execute.call_count, tool_count)
        self.assertEqual(len(model_inputs), tool_count + 1)
        self.assertEqual(self.service.session.task.status, 'answered')
        self.assertTrue(self.service.session.summary)
        self.assertLessEqual(estimate_tokens(self.service.session.summary), self.service.summary_tokens)
        self.assertTrue(any(json.loads(item['memory_context'])['summary'] for item in model_inputs))
        self.assertIn('Tool read_file src/file_', ' '.join(self.service.session.summary))
        self.assert_protocol_complete(self.service.session.context)
        self.assertLessEqual(estimate_tokens(self.service.session.context), 6000)
        final_tool = next(message for message in reversed(model_inputs[-1]['context']) if message['role'] == 'tool')
        self.assertEqual(final_tool['tool_call_id'], 'read-11')
        self.assertEqual(json.loads(final_tool['content'])['content'], '#' + 'a' * 3499)

    def test_large_memory_search_keeps_full_structured_response_in_next_request(self):
        saved = []
        for index in range(6):
            content = f'SQLite migration component {index}: ' + ('Keep migration history for audit. ' * 25)
            saved.append(self.service.remember(content[:650], 'requirement'))
        outputs = []
        original_execute = self.service.execute_tool
        seen = []

        def memory_tool(call):
            output = original_execute(call)
            outputs.append(output)
            return output

        def coder(text, **kwargs):
            seen.append(deepcopy(kwargs))
            if len(seen) == 1:
                return request_tool(kwargs['context'], tool_call('recall-1', 'memory_search', query='SQLite migration'))
            result = kwargs['context'][-1]
            self.assertEqual(result['role'], 'tool')
            self.assertEqual(result['tool_call_id'], 'recall-1')
            self.assertGreater(len(result['content']), 4000)
            self.assertEqual(result['content'], outputs[0])
            context = json.loads(result['content'])
            self.assertEqual(set(context), {'summary', 'task', 'memories', 'warning'})
            self.assertNotIn('excerpt', context)
            by_id = {record.id: record.content for record in saved}
            self.assertEqual(len(context['memories']), 6)
            self.assertTrue(all(entry['content'] == by_id[entry['id']] for entry in context['memories']))
            return completion(kwargs['context'])

        with patch('main.send_to_coder', side_effect=coder), \
                patch.object(self.service, 'execute_tool', side_effect=memory_tool), patch('builtins.print'):
            run_turn(self.service, 'Recall SQLite migration requirements',
                     sandbox_factory=lambda: self.fail('Memory search must not start a sandbox'))
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(outputs), 1)

    def test_source_expressions_survive_loop_but_literal_credentials_are_redacted(self):
        ordinary_source = ('import os\n\n'
                           'def validate(password: str) -> bool:\n'
                           '    password = input("Enter password: ")\n'
                           '    secret = os.environ["APP_SECRET"]\n'
                           '    api_key = os.getenv("API_KEY")\n'
                           '    return bool(password)\n')
        secret = 'synthetic_literal_password_value_123'
        source = ordinary_source + f'\nsaved_password = "{secret}"\n'
        requests = []

        def coder(text, **kwargs):
            requests.append(deepcopy(kwargs))
            if len(requests) == 1:
                return request_tool(kwargs['context'], tool_call('source-1', 'read_file', path='auth.py'))
            observation = json.loads(kwargs['context'][-1]['content'])
            self.assertTrue(observation['content'].startswith(ordinary_source))
            self.assertIn('saved_password = "[REDACTED]"', observation['content'])
            self.assertNotIn(secret, json.dumps(kwargs))
            compile(observation['content'], 'auth.py', 'exec')
            return completion(kwargs['context'])

        with patch('main.send_to_coder', side_effect=coder), \
                patch('main.execute_tool', return_value=json.dumps({'content': source, 'truncated': False})), \
                patch('builtins.print'):
            run_turn(self.service, 'Inspect authentication validation', sandbox_factory=Mock())
        self.assertEqual(len(requests), 2)
        with self.store._connect() as db:
            persisted = db.execute('SELECT context FROM memory_sessions WHERE session_id = ?',
                                   (self.service.session.session_id,)).fetchone()[0]
        self.assertNotIn(secret, persisted)


if __name__ == '__main__':
    unittest.main()

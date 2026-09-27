"""Regressions for renewed evidence, uniform budgets and runtime observations."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from main import run_turn
from memory import MemoryStore, get_relevant_memories
from memory.context import estimate_tokens
from memory.service import MemoryService
from scripts.tools.terminal_use import execute_command, OUTPUT_LIMIT


class StoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MemoryStore(Path(self.temp.name) / 'memory.db')

    def test_user_reconfirmation_promotes_existing_tool_fact_without_duplicate(self):
        expiry = datetime.now(timezone.utc) + timedelta(days=1)
        old = self.store.add('Use SQLite for persistence', 'requirement',
            project_id='p', user_id='u', source_type='tool', source_reference='earlier-tool',
            reliability=0.4, importance=0.2, valid_until=expiry)
        service = MemoryService(self.store, project_id='p', user_id='u')
        confirmed = service.remember('Use SQLite for persistence', 'requirement')
        self.assertEqual(confirmed.id, old.id)
        self.assertEqual(confirmed.created_at, old.created_at)
        self.assertEqual(confirmed.source_type, 'user')
        self.assertEqual(confirmed.reliability, 1.0)
        self.assertEqual(confirmed.importance, 0.85)
        self.assertIsNone(confirmed.valid_until)
        self.assertGreater(confirmed.updated_at, old.updated_at)
        envelope = json.loads(service.build_context('Implement endpoint'))
        self.assertIn(old.id, [record['id'] for record in envelope['memories']])
        self.assertEqual(len(service.visible_records()), 1)

    def test_repeated_tool_evidence_refreshes_expiry_recency_and_source(self):
        soon = datetime.now(timezone.utc) + timedelta(days=1)
        later = soon + timedelta(days=30)
        old = self.store.add('pytest returned exit code 1', 'error', source_type='tool',
            source_reference='run:old', reliability=0.95, valid_until=soon,
            metadata={'task_id': 'old'})
        new = self.store.add(old.content, 'error', source_type='tool',
            source_reference='run:new', reliability=0.95, valid_until=later,
            metadata={'task_id': 'new'})
        self.assertEqual(new.id, old.id)
        self.assertEqual(new.valid_until, later.isoformat())
        self.assertEqual(new.source_reference, 'run:new')
        self.assertEqual(new.metadata['task_id'], 'new')
        self.assertGreater(new.updated_at, old.updated_at)

    def test_weaker_echo_cannot_overwrite_user_provenance_or_ttl(self):
        expiry = datetime.now(timezone.utc) + timedelta(days=2)
        old = self.store.add('Use Python 3.12', source_type='user', source_reference='user:1',
                            reliability=0.9, valid_until=expiry)
        for source, confidence in [('tool', 1.0), ('model', 1.0), ('user', 0.5)]:
            with self.subTest(source=source, confidence=confidence):
                echo = self.store.add(old.content, source_type=source,
                    source_reference='weak-echo', reliability=confidence, valid_until=None)
                self.assertEqual(echo, old)

    def test_expired_new_evidence_cannot_refresh_a_current_fact(self):
        old = self.store.add('Use Python 3.12', reliability=0.9)
        stale = self.store.add(old.content, reliability=1.0,
            valid_until=datetime.now(timezone.utc) - timedelta(days=1))
        self.assertEqual(stale, old)

    def test_store_budget_uses_same_utf8_estimate_as_runtime(self):
        self.store.add('測試 ' * 100)
        self.assertEqual(self.store.retrieve('測試', max_tokens=100), [])
        hits = self.store.retrieve('測試', max_tokens=300)
        self.assertEqual(len(hits), 1)
        self.assertLessEqual(sum(estimate_tokens(hit.content) for hit in hits), 300)

    def test_legacy_budget_includes_metadata_and_plain_list_content(self):
        self.store.add('Python', source_reference='documentation/' + 'x' * 200)
        self.assertEqual(get_relevant_memories('Python', store=self.store, max_tokens=20), [])
        values = ['Python ' + '測試 ' * 100, 'Python 3.12']
        selected = get_relevant_memories('Python', memories=values, max_tokens=20)
        self.assertEqual(selected, ['Python 3.12'])
        self.assertLessEqual(estimate_tokens('\n'.join(selected)), 20)
        self.assertEqual(get_relevant_memories('Python', memories=values, limit=-1), [])

    def test_restored_environment_notice_reaches_model_outside_small_memory_budget(self):
        original = MemoryService(self.store, project_id='p', user_id='u')
        original.start_task('Implement API')
        original.update_task(artifacts=['app.py'])
        restored = MemoryService(self.store, project_id='p', user_id='u', memory_tokens=64,
                                 session_id=original.session.session_id)
        response = Mock()
        response.json.return_value = {'choices': [{'message': {
            'role': 'assistant', 'content': 'I need to inspect current files.'}, 'finish_reason': 'stop'}]}
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test'}), \
                patch('agent_roles.coder.requests.post', return_value=response) as post, patch('builtins.print'):
            run_turn(restored, 'Continue', sandbox_factory=lambda: self.fail('No replay'))
        messages = post.call_args.kwargs['json']['messages']
        self.assertIn({'role': 'system', 'content': restored.environment_notice}, messages)
        self.assertIn('not a sandbox snapshot', restored.environment_notice)
        self.assertFalse(any(m['role'] == 'system' for m in restored.session.context))


class TerminalObservationAuditTests(unittest.TestCase):
    def test_long_output_preserves_diagnostic_tail_in_memory(self):
        result = SimpleNamespace(stdout='START\n' + 'progress\n' * 5000 + 'EADDRINUSE port 8000',
                                 stderr='', exit_code=1)
        with patch('scripts.tools.terminal_use.run_command', return_value=result):
            output = execute_command({'command': 'pytest'}, object())
        self.assertLessEqual(len(output['stdout']), OUTPUT_LIMIT)
        self.assertTrue(output['stdout'].startswith('START'))
        self.assertTrue(output['stdout'].endswith('EADDRINUSE port 8000'))
        self.assertTrue(output['truncated'])
        with tempfile.TemporaryDirectory() as directory:
            service = MemoryService(MemoryStore(Path(directory) / 'm.db'), project_id='p', user_id='u')
            service.start_task('Fix tests')
            service.observe_tool({'id': 'test-1', 'function': {'name': 'run_command',
                'arguments': '{"command":"pytest"}'}}, json.dumps(output))
            self.assertIn('EADDRINUSE', service.recall_query())

    def test_complete_long_credential_is_filtered_before_output_clipping(self):
        secret = 'SYNTHETIC_PRIVATE_VALUE_' * 1000
        stdout = 'password = "' + secret + '"\nLAST_LINE'
        result = SimpleNamespace(stdout=stdout, stderr='', exit_code=1)
        with patch('scripts.tools.terminal_use.run_command', return_value=result):
            output = execute_command({'command': 'inspect-config'}, object())
        self.assertNotIn('SYNTHETIC_PRIVATE_VALUE', json.dumps(output))
        self.assertIn('REDACTED', output['stdout'])
        self.assertIn('LAST_LINE', output['stdout'])
        self.assertTrue(output['redacted'])


if __name__ == '__main__':
    unittest.main()

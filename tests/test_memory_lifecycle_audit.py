"""Behavioral regressions found by the second independent lifecycle audit."""
import json
from pathlib import Path
import tempfile
import unittest

from memory import MemoryStore
from memory.prompt import encode_memory_envelope
from memory.service import MemoryService, estimate_tokens


def remember_call(quote):
    return {'function': {'name': 'memory_remember', 'arguments': json.dumps({
        'source_quote': quote, 'memory_type': 'requirement',
    })}}


class LifecycleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.db'
        self.store = MemoryStore(self.path)
        self.service = MemoryService(self.store, project_id='project', user_id='alice')

    def test_coding_verb_does_not_block_independent_requirement(self):
        self.service.remember('Use SQLite for persistence', 'requirement')
        self.service.start_task('Replace deprecated imports. Use pytest for all tests.')
        result = json.loads(self.service.execute_tool(remember_call('Use pytest for all tests.')))
        self.assertIn('saved_memory_id', result)
        self.assertEqual(self.store.get(result['saved_memory_id']).content, 'Use pytest for all tests.')

    def test_first_fact_with_change_word_has_no_required_predecessor(self):
        quote = 'Remember switch to Python 3.12'
        self.service.start_task(quote)
        saved = self.service.observe_user(quote)
        self.assertIsNotNone(saved)
        result = json.loads(self.service.execute_tool(remember_call(quote)))
        # Duplicate promotion of the same current-user fact remains idempotent.
        self.assertEqual(result['saved_memory_id'], saved.id)

    def test_unrelated_memory_is_not_predecessor_based_on_boilerplate(self):
        old = self.service.remember('Remember use SQLite', 'requirement')
        quote = 'Remember switch to Python 3.12'
        self.service.start_task(quote)
        result = json.loads(self.service.execute_tool(remember_call(quote)))
        self.assertIn('saved_memory_id', result)
        self.assertEqual(self.store.get(old.id).status, 'active')

    def test_changed_fact_still_requires_explicit_predecessor(self):
        old = self.service.remember('Use SQLite for persistence', 'requirement')
        quote = 'Use PostgreSQL for persistence instead of SQLite'
        self.service.start_task(quote)
        result = json.loads(self.service.execute_tool(remember_call(quote)))
        self.assertIn('error', result)
        self.assertIn(str(old.id), result['error'])
        self.assertEqual(len(self.service.visible_records()), 1)

    def test_expired_or_foreign_facts_do_not_prevent_new_fact(self):
        self.store.add('Use SQLite for persistence', 'requirement', project_id='other',
                       user_id='alice', source_type='user')
        expired = self.service.remember('Use SQLite for persistence', 'requirement')
        self.store.expire(expired.id)
        quote = 'Use PostgreSQL for persistence instead of SQLite'
        self.service.start_task(quote)
        result = json.loads(self.service.execute_tool(remember_call(quote)))
        self.assertIn('saved_memory_id', result)

    def test_standing_requirements_are_packed_beyond_arbitrary_four(self):
        facts = ['Python 3.12', 'SQLite persistence', 'Offline operation',
                 'Pytest validation', 'Preserve public compatibility', 'No paid services']
        ids = {self.service.remember(fact, 'requirement').id for fact in facts}
        self.service.start_task('Implement the endpoint')
        envelope = json.loads(self.service.build_context('Implement the endpoint'))
        self.assertEqual({entry['id'] for entry in envelope['memories']}, ids)
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], envelope)), 1800)

    def test_many_standing_facts_still_fit_small_memory_budget(self):
        for index in range(70):
            self.service.remember(f'Requirement {index}: preserve the API contract', 'requirement')
        service = MemoryService(self.store, project_id='project', user_id='alice', memory_tokens=250)
        envelope = json.loads(service.build_context('unrelated'))
        self.assertGreater(len(envelope['memories']), 0)
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], envelope)), 250)

    def test_standing_requirements_cannot_starve_relevant_error_lesson(self):
        for index in range(30):
            self.service.remember(
                f'Requirement {index}: keep the public request and response contracts stable '
                'across released API versions. Document backwards incompatible changes before shipping.',
                'requirement')
        lesson = self.store.add(
            'EADDRINUSE means port 8000 is already bound; inspect the owning process before retrying.',
            'lesson', project_id='project', user_id='alice', source_type='tool',
            reliability=0.95, importance=0.75)
        self.service.start_task('Fix failing server')
        envelope = json.loads(self.service.build_context('EADDRINUSE port 8000'))
        self.assertIn(lesson.id, [entry['id'] for entry in envelope['memories']])
        self.assertTrue(any(entry['type'] == 'requirement' for entry in envelope['memories']))
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], envelope)), 1800)
        self.assertGreater(len(envelope['memories']), 4)

    def test_diagnostic_tail_survives_each_working_memory_excerpt(self):
        self.service.start_task('Fix server')
        call = {'id': 'run-1', 'function': {'name': 'run_command',
                                          'arguments': '{"command":"python server.py"}'}}
        self.service.observe_tool(call, json.dumps({
            'exit_code': 1, 'stderr': 'START\n' + 'progress\n' * 500 + 'EADDRINUSE port 8000',
        }))
        pending = next(iter(self.service.session.task.pending_failures.values()))
        self.assertTrue(pending['error'].endswith('EADDRINUSE port 8000'))
        self.assertLessEqual(len(pending['error']), 500)
        self.assertIn('EADDRINUSE', self.service.recall_query())
        envelope = json.loads(self.service.build_context('server'))
        output = envelope['task']['latest_observation']['output_excerpt']
        self.assertTrue(output.endswith('EADDRINUSE port 8000'))
        self.assertLessEqual(len(output), 240)

    def test_restore_notice_survives_continuation_and_new_clears_it(self):
        self.service.start_task('Implement storage')
        self.service.update_task(artifacts=['storage.py'])
        resumed = MemoryService(MemoryStore(self.path), project_id='project', user_id='alice',
                                session_id=self.service.session.session_id)
        self.assertTrue(resumed.restored_session)
        self.assertIn('not a sandbox snapshot', resumed.environment_notice)
        resumed.begin_turn('Continue')
        self.assertIsNone(resumed.session.task.error)
        self.assertIn('different sandbox', resumed.environment_notice)
        envelope = json.loads(resumed.build_context('storage'))
        self.assertEqual(envelope['task']['environment_notice'], resumed.environment_notice)
        session_id = resumed.session.session_id
        resumed.switch_session()
        self.assertFalse(resumed.restored_session)
        self.assertIsNone(resumed.environment_notice)
        resumed.switch_session(session_id)
        self.assertTrue(resumed.restored_session)


if __name__ == '__main__':
    unittest.main()

"""Realistic regressions from the practical memory audit (all offline)."""
import json
from pathlib import Path
import tempfile
import unittest

from memory import MemoryStore
from memory.context import ContextBudgetExceeded, compact_history, estimate_tokens
from memory.prompt import encode_memory_envelope
from memory.service import CheckpointConflict, MemoryService


def call(name, identifier='call-1', **arguments):
    return {'id': identifier, 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(arguments),
    }}


class MemoryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.db'
        self.store = MemoryStore(self.path)
        self.service = MemoryService(self.store, project_id='project', user_id='alice')

    def resume(self):
        return MemoryService(MemoryStore(self.path), project_id='project', user_id='alice',
                             session_id=self.service.session.session_id)

    def test_large_plan_cannot_starve_relevant_requirement(self):
        required = self.service.remember('Use pytest for validation', 'requirement')
        self.service.start_task('Implement the API')
        self.service.update_task(current_plan=[str(i) + 'a' * 480 for i in range(12)],
                                 active_step='b' * 480)
        context = json.loads(self.service.build_context('pytest'))
        self.assertIn(required.id, [m['id'] for m in context['memories']])
        self.assertIsNotNone(context['task'])
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], context)), 1800)

    def test_prompt_keeps_facts_while_diagnostics_stay_in_trace(self):
        for fact in ('Use Python 3.12', 'Use SQLite', 'Run pytest', 'Keep API stable'):
            self.service.remember(fact, 'requirement')
        self.service.start_task('Implement API')
        context = json.loads(self.service.build_context('Implement API'))
        self.assertEqual(len(context['memories']), 4)
        self.assertLess(estimate_tokens(encode_memory_envelope([], context)), 330)
        for fact in context['memories']:
            self.assertNotIn('why', fact)
            self.assertNotIn('score', fact)
            self.assertNotIn('session:', fact['source'])
        self.assertTrue(all('why' in fact for fact in self.service.last_retrieval['matches']))
        self.assertNotIn('pending_failures', context['task'])

    def test_large_requirement_can_reclaim_unused_working_state_budget(self):
        requirement = self.service.remember('Required: ' + 'detail ' * 520, 'requirement')
        self.service.start_task('Implement API ' + 'goal ' * 80)
        self.service.update_task(current_plan=['step ' * 95 for _ in range(12)],
            constraints=['constraint ' * 40 for _ in range(6)],
            artifacts=['path/' + 'a' * 470 for _ in range(6)])
        context = json.loads(self.service.build_context('Required detail'))
        self.assertIn(requirement.id, [m['id'] for m in context['memories']])
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], context)), 1800)

    def test_failure_read_edit_restart_retest_creates_evidence_linked_lesson(self):
        self.service.start_task('Fix failing validation')
        self.service.observe_tool(call('run_command', command='pytest -q'),
                                  json.dumps({'exit_code': 1, 'stderr': 'FAILED test_validation'}))
        self.service.observe_tool(call('read_file', path='app.py'), '{"content":"old source"}')
        self.service.observe_tool(call('write_file', path='app.py', content='new source'),
                                  '{"written":true}')
        self.service.checkpoint()
        resumed = self.resume()
        resumed.begin_turn('Continue testing the change')
        resumed.observe_tool(call('run_command', command='pytest -q'),
                             '{"exit_code":0,"stdout":"3 passed"}')
        resumed.checkpoint()
        lessons = [m for m in resumed.visible_records() if m.memory_type == 'lesson']
        self.assertEqual(len(lessons), 1)
        evidence = json.loads(lessons[0].content)
        self.assertEqual(evidence['changed_files'], ['app.py'])
        self.assertEqual(evidence['previous_error'], 'FAILED test_validation')
        self.assertEqual(self.store.get(evidence['failure_memory_id']).memory_type, 'error')
        self.assertIn('not established', evidence['limitation'])
        self.assertEqual(resumed.session.task.pending_failures, {})
        self.assertIn('app.py', resumed.session.task.artifacts)

    def test_new_task_cannot_pair_success_with_prior_task_failure(self):
        self.service.start_task('Earlier task')
        self.service.observe_tool(call('run_command', command='pytest'), '{"exit_code":1}')
        self.service.begin_turn('Separate task', new_task=True)
        self.service.observe_tool(call('run_command', command='pytest'), '{"exit_code":0}')
        self.assertFalse(any(m.memory_type == 'lesson' for m in self.service.visible_records()))

    def test_failed_write_is_not_a_recovery_artifact(self):
        self.service.start_task('Fix tests')
        self.service.observe_tool(call('run_command', command='pytest'), '{"exit_code":1}')
        self.service.observe_tool(call('write_file', path='app.py', content='new source'),
                                  '{"error":"Permission denied"}')
        self.service.observe_tool(call('run_command', command='pytest'), '{"exit_code":0}')
        lesson = next(m for m in self.service.visible_records() if m.memory_type == 'lesson')
        self.assertEqual(json.loads(lesson.content)['changed_files'], [])
        self.assertEqual(self.service.session.task.artifacts, [])

    def test_grounded_correction_uses_current_continuation_and_supersedes(self):
        previous = self.service.observe_user('Remember use SQLite exclusively for persistence')
        self.service.start_task('Implement persistent storage')
        quote = 'Remember use PostgreSQL exclusively instead of SQLite'
        self.service.begin_turn(quote)
        self.assertIsNone(self.service.observe_user(quote))
        duplicate = json.loads(self.service.execute_tool(call('memory_remember',
            source_quote=quote, memory_type='requirement')))
        self.assertIn('error', duplicate)
        result = json.loads(self.service.execute_tool(call('memory_correct',
            memory_id=previous.id, source_quote=quote)))
        self.assertEqual(result['supersedes_id'], previous.id)
        self.assertEqual(self.store.get(previous.id).status, 'superseded')
        context = json.loads(self.service.build_context('persistence'))
        self.assertNotIn(previous.id, [m['id'] for m in context['memories']])
        self.assertIn(result['saved_memory_id'], [m['id'] for m in context['memories']])

    def test_correction_rejects_invented_quotes_foreign_scope_and_boolean_ids(self):
        foreign = self.store.add('Use SQLite', project_id='other', user_id='alice')
        self.service.start_task('Use PostgreSQL for persistence')
        local = self.service.remember('Use SQLite')
        for memory_id, quote in ((local.id, 'Use invented MySQL'),
                                 (foreign.id, 'Use PostgreSQL for persistence'),
                                 (True, 'Use PostgreSQL for persistence')):
            with self.subTest(memory_id=memory_id, quote=quote):
                result = json.loads(self.service.execute_tool(call('memory_correct',
                    memory_id=memory_id, source_quote=quote)))
                self.assertIn('error', result)
        self.assertEqual(self.store.get(local.id).status, 'active')

    def test_continuation_write_cannot_quote_previous_user_as_current(self):
        self.service.start_task('Use SQLite for persistence')
        self.service.begin_turn('Use pytest for validation')
        old = json.loads(self.service.execute_tool(call('memory_remember',
            source_quote='Use SQLite for persistence', memory_type='requirement')))
        new = json.loads(self.service.execute_tool(call('memory_remember',
            source_quote='Use pytest for validation', memory_type='requirement')))
        self.assertIn('error', old)
        self.assertIn('saved_memory_id', new)

    def test_conflicting_continuation_rolls_back_local_task(self):
        self.service.start_task('Original task')
        stale = self.resume()
        previous = stale.session.task
        self.service.update_task(active_step='Authoritative step')
        with self.assertRaises(CheckpointConflict):
            stale.begin_turn('Continue')
        self.assertEqual(stale.session.task, previous)
        self.assertNotEqual(stale.current_user_request(), 'Continue')

    def test_answered_task_has_explicit_continuation_and_new_task_semantics(self):
        original = self.service.start_task('Build API').task_id
        self.service.finish_task('answered')
        self.service.begin_turn('Also validate the input', continue_task=True)
        self.assertEqual(self.service.session.task.task_id, original)
        self.service.finish_task('answered')
        self.service.begin_turn('Build unrelated report')
        self.assertNotEqual(self.service.session.task.task_id, original)

    def test_recall_uses_active_error_and_step(self):
        self.service.start_task('Build a server')
        self.service.update_task(active_step='Start integration server')
        self.service.observe_tool(call('run_command', command='python server.py'),
                                  '{"exit_code":1,"stderr":"EADDRINUSE port 8000"}')
        query = self.service.recall_query('Continue')
        self.assertIn('EADDRINUSE', query)
        self.assertIn('Start integration server', query)
        self.assertIn('Build a server', query)
        self.service.observe_tool(call('read_file', path='server.py'), '{"content":"source"}')
        self.assertIn('EADDRINUSE', self.service.recall_query('Continue'))
        self.service.observe_tool(call('run_command', command='python server.py'), '{"exit_code":0}')
        self.assertNotIn('EADDRINUSE', self.service.recall_query('Continue'))

    def test_compaction_preserves_parallel_pending_batch_and_summary(self):
        old = call('read_file', 'old', path='old.py')
        first, second = call('read_file', 'first', path='a.py'), call('read_file', 'second', path='b.py')
        history = [
            {'role': 'user', 'content': 'Inspect source'},
            {'role': 'assistant', 'content': '', 'tool_calls': [old]},
            {'role': 'tool', 'tool_call_id': 'old', 'content': json.dumps({'content': 'x' * 2500})},
            {'role': 'assistant', 'content': '', 'tool_calls': [first, second]},
            {'role': 'tool', 'tool_call_id': 'first', 'content': '{"content":"a"}'},
        ]
        compacted, summary = compact_history(history, [], history_tokens=400, summary_tokens=150)
        self.assertEqual(compacted, [history[0], *history[-2:]])
        self.assertIn('old.py', ' '.join(summary))
        compacted.append({'role': 'tool', 'tool_call_id': 'second', 'content': '{"content":"b"}'})
        checked, _ = compact_history(compacted, summary, history_tokens=400, summary_tokens=150)
        self.assertEqual(compacted, checked)

    def test_oversized_pending_exchange_fails_without_slicing_protocol(self):
        history = [{'role': 'user', 'content': 'Inspect source'},
                   {'role': 'assistant', 'content': '', 'tool_calls': [
                       call('write_file', content='x' * 2500, path='file.py')]}]
        with self.assertRaises(ContextBudgetExceeded):
            compact_history(history, [], history_tokens=400, summary_tokens=150)


if __name__ == '__main__':
    unittest.main()

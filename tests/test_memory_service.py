"""Behavioral checks for persisted sessions, write policy and agent integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memory import MemoryStore
from memory.service import (CheckpointConflict, ContextBudgetExceeded, MemoryService,
                            estimate_tokens, project_identity)
from main import finish_failed_task, memory_command, run_turn
from memory.prompt import encode_memory_envelope
from scripts.tools.sandbox_setup import project_files


def call(name, **arguments):
    return {'id': 'call-1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'memory.db'
        self.store = MemoryStore(self.path)
        self.service = MemoryService(self.store, project_id='project', user_id='alice')

    def tearDown(self):
        self.temp.cleanup()

    def service_for(self, **kwargs):
        options = {'project_id': 'project', 'user_id': 'alice'}
        options.update(kwargs)
        return MemoryService(MemoryStore(self.path), **options)

    def test_restart_restores_history_task_and_long_term_fact(self):
        saved = self.service.remember('Use Python 3.12', 'requirement')
        task = self.service.start_task('Implement the Python API')
        self.service.checkpoint(self.service.session.context + [{'role': 'assistant', 'content': 'I will inspect the API.'}])
        resumed = self.service_for(session_id=self.service.session.session_id)
        self.assertEqual(resumed.session.task.task_id, task.task_id)
        self.assertEqual(resumed.session.task.status, 'interrupted')
        self.assertEqual(len(resumed.session.context), 2)
        self.assertIn(saved.id, [m['id'] for m in json.loads(resumed.build_context('Python'))['memories']])

    def test_new_session_recalls_facts_but_not_other_session_history(self):
        self.service.remember('Database uses SQLite')
        self.service.start_task('Private conversation')
        fresh = self.service_for()
        self.assertEqual(fresh.session.context, [])
        self.assertEqual(len(json.loads(fresh.build_context('SQLite'))['memories']), 1)

    def test_other_user_cannot_load_or_mutate_memory(self):
        record = self.service.remember('Database uses SQLite')
        other = self.service_for(user_id='bob')
        self.assertEqual(other.visible_records(), [])
        with self.assertRaises(KeyError):
            other.forget(record.id)
        with self.assertRaises(KeyError):
            other.correct(record.id, 'Use another database')
        with self.assertRaises(KeyError):
            self.service_for(user_id='bob', session_id=self.service.session.session_id)

    def test_project_identity_uses_full_path(self):
        self.assertNotEqual(project_identity(Path('/one/project')), project_identity(Path('/two/project')))

    def test_compaction_preserves_whole_tool_exchange_and_bounded_summary(self):
        service = self.service_for(history_tokens=250, summary_tokens=100)
        transcript = [{'role': 'user', 'content': 'Old goal ' + 'a' * 450}, {'role': 'assistant', 'content': 'Old answer'}]
        transcript += [{'role': 'user', 'content': 'Current goal'}, {'role': 'assistant', 'content': None, 'tool_calls': [call('list_files', path='.')]}, {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"files":[]}'}, {'role': 'assistant', 'content': 'Done'}]
        service.checkpoint(transcript)
        self.assertEqual([m['role'] for m in service.session.context], ['user', 'assistant', 'tool', 'assistant'])
        self.assertLessEqual(estimate_tokens(service.session.context), 250)
        self.assertLessEqual(estimate_tokens(service.session.summary), 100)

    def test_active_turn_overflow_is_explicit_and_keeps_previous_checkpoint(self):
        service = self.service_for(history_tokens=100)
        service.checkpoint([{'role': 'user', 'content': 'small'}])
        with self.assertRaises(ContextBudgetExceeded):
            service.checkpoint([{'role': 'user', 'content': 'large' * 200}])
        self.assertEqual(service.session.context[0]['content'], 'small')

    def test_partial_tool_batch_resume_adds_unknown_result_without_replay(self):
        first, second = call('run_command', command='touch x'), call('run_command', command='touch y')
        second['id'] = 'call-2'
        self.service.start_task('Create files')
        self.service.checkpoint(self.service.session.context + [{'role': 'assistant', 'content': None, 'tool_calls': [first, second]}, {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"exit_code":0}'}])
        resumed = self.service_for(session_id=self.service.session.session_id)
        tools = [m for m in resumed.session.context if m['role'] == 'tool']
        self.assertEqual([m['tool_call_id'] for m in tools], ['call-1', 'call-2'])
        self.assertIn('not replayed', tools[-1]['content'])

    def test_stale_worker_cannot_overwrite_new_checkpoint(self):
        stale = self.service_for(session_id=self.service.session.session_id)
        self.service.start_task('Newest goal')
        with self.assertRaises(CheckpointConflict):
            stale.checkpoint([{'role': 'user', 'content': 'Stale goal'}])
        restored = self.service_for(session_id=self.service.session.session_id)
        self.assertEqual(restored.session.task.user_goal, 'Newest goal')

    def test_failed_task_updates_restore_local_state(self):
        self.service.start_task('Original goal')
        stale = self.service_for(session_id=self.service.session.session_id)
        self.service.update_task(current_plan=['Authoritative plan'])
        previous = stale.session.task
        with self.assertRaises(CheckpointConflict):
            stale.update_task(current_plan=['Never persisted'])
        self.assertEqual(stale.session.task, previous)
        with self.assertRaises(CheckpointConflict):
            stale.finish_task('answered')
        self.assertEqual(stale.session.task, previous)

    def test_current_user_correction_renews_elapsed_ttl(self):
        record = self.service.remember('Use old Python version', 'requirement')
        self.store.update(record.id, valid_until='2000-01-01T00:00:00+00:00')
        corrected = self.service.correct(record.id, 'Use Python 3.13')
        self.assertIsNone(corrected.valid_until)
        self.assertIn(corrected.id, [m['id'] for m in json.loads(self.service.build_context('Python'))['memories']])

    def test_no_reasoning_or_credentials_in_checkpoint(self):
        secret = 'sk-or-v1-' + 'x' * 40
        self.service.checkpoint([{'role': 'user', 'content': f'api_key="{secret}"'}, {'role': 'assistant', 'content': 'ok', 'reasoning': 'private scratchpad', 'reasoning_details': [{'text': 'hidden'}]}])
        with self.store._connect() as db:
            stored = db.execute('SELECT context FROM memory_sessions WHERE session_id = ?', (self.service.session.session_id,)).fetchone()[0]
        self.assertNotIn(secret, stored)
        self.assertNotIn('scratchpad', stored)
        self.assertNotIn('reasoning', stored)
        self.assertIn('REDACTED', stored)

    def test_json_encoded_tool_source_credentials_are_redacted_and_stay_valid(self):
        secret = 'dummy_sensitive_value_987'
        source = f'password = "{secret}"'
        command = call('write_file', path='config.py', content=source)
        cleaned = self.service.clean_context([
            {'role': 'assistant', 'content': '', 'tool_calls': [command]},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': json.dumps({'content': source})},
        ])
        self.assertNotIn(secret, json.dumps(cleaned))
        self.assertIn('REDACTED', json.loads(cleaned[0]['tool_calls'][0]['function']['arguments'])['content'])
        self.assertIn('REDACTED', json.loads(cleaned[1]['content'])['content'])
        from memory import SensitiveMemoryError
        with self.assertRaises(SensitiveMemoryError):
            self.store.add(json.dumps({'content': source}))

    def test_selective_user_write_and_grounded_memory_tool(self):
        self.assertIsNone(self.service.observe_user('Hello there!'))
        self.assertIsNone(self.service.observe_user('tui muốn sửa function này'))
        self.assertIsNotNone(self.service.observe_user('I prefer pytest for testing'))
        self.service.start_task('The API must use SQLite and support pagination.')
        saved = json.loads(self.service.execute_tool(call('memory_remember', source_quote='The API must use SQLite', memory_type='requirement')))
        self.assertIn('saved_memory_id', saved)
        rejected = json.loads(self.service.execute_tool(call('memory_remember', source_quote='The API must use PostgreSQL', memory_type='requirement')))
        self.assertIn('error', rejected)

    def test_memory_tool_rejects_scope_injection(self):
        output = json.loads(self.service.execute_tool(call('memory_search', query='secret', user_id='bob')))
        self.assertIn('error', output)

    def test_agent_can_checkpoint_plan_without_promoting_it_to_facts(self):
        self.service.start_task('Build API')
        result = json.loads(self.service.execute_tool(call('memory_update_task', current_plan=['Inspect routes', 'Implement handler', 'Test'], active_step='Inspect routes')))
        self.assertIn('checkpoint_revision', result)
        resumed = self.service_for(session_id=self.service.session.session_id)
        self.assertEqual(resumed.session.task.current_plan[0], 'Inspect routes')
        self.assertEqual(self.service.visible_records(), [])

    def test_tool_observation_is_bounded_scoped_and_expires(self):
        self.service.start_task('Fix pytest')
        self.service.observe_tool(call('run_command', command='pytest'), json.dumps({'exit_code': 1, 'stderr': 'error ' * 2000}))
        record = self.service.visible_records()[0]
        self.assertEqual(record.memory_type, 'error')
        self.assertEqual(record.source_type, 'tool')
        self.assertIsNotNone(record.valid_until)
        self.assertLess(len(record.content), 1800)
        self.assertEqual(self.service.session.task.retry_count, 1)

    def test_file_contents_not_promoted_to_durable_memory(self):
        self.service.observe_tool(call('read_file', path='main.py'), json.dumps({'content': 'entire file'}))
        self.assertEqual(self.service.visible_records(), [])

    def test_large_private_key_is_redacted_before_tool_excerpt(self):
        body = 'SYNTHETIC_PRIVATE_BODY_' * 150
        output = json.dumps({'exit_code': 1, 'stderr': f'-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----'})
        self.service.start_task('Diagnose error')
        self.service.observe_tool(call('run_command', command='pytest'), output)
        self.service.checkpoint()
        durable = ' '.join(r.content for r in self.service.visible_records())
        self.assertNotIn('SYNTHETIC_PRIVATE_BODY', durable)
        self.assertNotIn('SYNTHETIC_PRIVATE_BODY', self.service.session.task.latest_observation)
        from main import bounded_tool_output
        from memory.service import redact
        self.assertNotIn('SYNTHETIC_PRIVATE_BODY', bounded_tool_output(redact(output)))

    def test_prompt_budget_includes_metadata_and_task(self):
        service = self.service_for(memory_tokens=300)
        service.remember('Python ' + 'large ' * 500)
        service.remember('Python 3.12')
        service.start_task('Build Python API')
        context = service.build_context('Python')
        self.assertLessEqual(estimate_tokens(context), 300)
        self.assertEqual(service.last_retrieval['estimated_tokens'], estimate_tokens(encode_memory_envelope([], json.loads(context))))

    def test_markup_expansion_cannot_exceed_injected_prompt_budget(self):
        service = self.service_for(memory_tokens=400)
        service.remember('Markup ' + '<&>' * 700, 'requirement')
        envelope = json.loads(service.build_context('Markup'))
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], envelope)), 400)

    def test_smallest_memory_budget_and_long_task_fit_final_envelope(self):
        service = self.service_for(memory_tokens=64)
        service.start_task('Long task ' + '<' * 600)
        envelope = json.loads(service.build_context('task'))
        self.assertLessEqual(estimate_tokens(encode_memory_envelope([], envelope)), 64)

    def test_standing_requirement_survives_no_keyword_overlap(self):
        record = self.service.remember('Use Python 3.12', 'requirement')
        envelope = json.loads(self.service.build_context('Implement the endpoint'))
        self.assertIn(record.id, [m['id'] for m in envelope['memories']])

    def test_shared_blackboard_plan_is_persistent_and_validated(self):
        self.service.start_task('Implement API')
        self.service.update_task(current_plan=['Inspect files', 'Implement', 'Run tests'], assigned_agent='reviewer', active_step='Run tests')
        resumed = self.service_for(session_id=self.service.session.session_id)
        self.assertEqual(resumed.session.task.assigned_agent, 'reviewer')
        self.assertEqual(resumed.session.task.active_step, 'Run tests')
        with self.assertRaises(ValueError):
            self.service.update_task(task_id='change-identity')

    def test_same_process_failure_closes_pending_tools_before_next_user(self):
        self.service.start_task('Run command')
        self.service.checkpoint(self.service.session.context + [{'role': 'assistant', 'content': '', 'tool_calls': [call('run_command', command='touch file')]}])
        self.service.finish_task('failed', 'Sandbox failed')
        self.service.start_task('Try again')
        roles = [m['role'] for m in self.service.session.context]
        self.assertEqual(roles, ['user', 'assistant', 'tool', 'user'])
        self.assertIn('unknown', self.service.session.context[2]['content'])

    def test_checkpoint_conflict_finalization_does_not_raise(self):
        stale = self.service_for(session_id=self.service.session.session_id)
        self.service.start_task('New goal')
        with self.assertRaises(CheckpointConflict):
            stale.start_task('Conflicting goal')
        self.assertIsNone(stale.session.task)
        with patch.object(stale, 'finish_task', side_effect=CheckpointConflict('conflict')), patch('builtins.print') as output:
            finish_failed_task(stale, 'failed', 'CheckpointConflict')
        self.assertIn('/resume', output.call_args.args[0])

    def test_custom_database_extension_is_excluded_from_sandbox(self):
        root = Path(self.temp.name)
        custom = root / 'memory.data'
        custom.write_text('private')
        self.assertNotIn(custom, list(project_files(root, excluded_paths=[custom])))

    def test_all_credential_formats_are_redacted_in_history_and_task(self):
        aws = 'AKIA' + 'A' * 16
        self.service.start_task(f'Handle {aws} refresh_token=shortsecret')
        self.service.checkpoint(self.service.session.context + [{'role': 'tool', 'tool_call_id': 'example', 'content': '{"private_key":"shortprivate","refresh_token":"shortrefresh"}'}])
        with self.store._connect() as db:
            row = db.execute('SELECT context, task FROM memory_sessions WHERE session_id = ?', (self.service.session.session_id,)).fetchone()
        for secret in (aws, 'shortsecret', 'shortrefresh', 'shortprivate'):
            self.assertNotIn(secret, ' '.join(row))

    def test_retrieval_failure_leaves_current_task_usable(self):
        import sqlite3
        self.service.start_task('Build API')
        with patch.object(self.store, 'retrieve', side_effect=sqlite3.OperationalError('broken index')):
            context = json.loads(self.service.build_context('API'))
        self.assertEqual(context['task']['user_goal'], 'Build API')
        self.assertEqual(context['memories'], [])
        self.assertIsNotNone(context['warning'])

    def test_delete_session_erases_transcript_and_events_only(self):
        self.service.remember('Durable SQLite requirement')
        self.service.start_task('Old goal')
        old = self.service.session.session_id
        self.service.switch_session()
        self.assertTrue(self.service.delete_session(old))
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM memory_events WHERE session_id = ?', (old,)).fetchone()[0], 0)
        self.assertEqual(len(self.service.visible_records()), 1)

    def test_cli_commands_do_not_require_sandbox_or_llm(self):
        with patch('builtins.print'), patch('main.send_to_coder') as coder:
            self.assertTrue(memory_command(self.service, '/remember Use SQLite'))
            self.assertTrue(memory_command(self.service, '/search SQLite'))
            coder.assert_not_called()
        self.assertEqual(len(self.service.visible_records()), 1)

    def test_memory_database_not_copied_to_agent_sandbox(self):
        root = Path(self.temp.name)
        (root / 'code.py').write_text('print(1)')
        (root / 'memory.db-wal').write_text('private')
        names = [p.name for p in project_files(root)]
        self.assertIn('code.py', names)
        self.assertNotIn('memory.db', names)
        self.assertNotIn('memory.db-wal', names)

    def test_full_turn_tool_failure_recovery_and_memory_next_session(self):
        self.service.remember('Use pytest to test the Python API', 'requirement')
        calls = []
        def coder(text, **kwargs):
            self.assertIsNone(text)
            calls.append(kwargs)
            context = list(kwargs['context'])
            if len(calls) <= 2:
                tool = call('run_command', command='pytest')
                tool['id'] = f'call-{len(calls)}'
                context.append({'role': 'assistant', 'content': None, 'tool_calls': [tool]})
                return {'has_tool_call': True, 'tool_calls': [tool], 'context': context, 'finish_reason': 'tool_calls'}
            context.append({'role': 'assistant', 'content': 'Tests passed after correction.'})
            return {'has_tool_call': False, 'tool_calls': [], 'context': context, 'text': 'Tests passed after correction.', 'finish_reason': 'stop'}
        with patch('main.send_to_coder', side_effect=coder), patch('main.execute_tool', side_effect=[json.dumps({'exit_code': 1, 'stderr': 'AssertionError'}), json.dumps({'exit_code': 0, 'stdout': '3 passed'})]), patch('builtins.print'):
            answer = run_turn(self.service, 'Fix the Python API tests', sandbox_factory=lambda: object())
        self.assertIn('Tests passed', answer)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.service.session.task.status, 'answered')
        self.assertTrue(all('pytest' in c['memory_context'] for c in calls))
        fresh = self.service_for()
        recalled = json.loads(fresh.build_context('pytest'))
        self.assertTrue(any(m['type'] == 'error' for m in recalled['memories']))
        self.assertTrue(any(m['type'] == 'lesson' for m in recalled['memories']))

    def test_model_loop_stops_at_configured_bound(self):
        def coder(text, **kwargs):
            tool = call('memory_search', query='SQLite')
            return {'has_tool_call': True, 'tool_calls': [tool], 'context': kwargs['context'] + [{'role': 'assistant', 'content': None, 'tool_calls': [tool]}]}
        with patch('main.send_to_coder', side_effect=coder), patch('builtins.print'):
            answer = run_turn(self.service, 'Search memory', sandbox_factory=lambda: self.fail('memory tools need no sandbox'), max_rounds=2)
        self.assertIn('Stopped after 2', answer)
        self.assertEqual(self.service.session.task.status, 'stopped')


if __name__ == '__main__':
    unittest.main()

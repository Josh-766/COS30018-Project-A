"""Offline regression tests for recoverable, bounded source/tool observations."""

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from memory.observations import bounded_tool_output
from memory.privacy import redact_sensitive_text
from scripts.tools.read_file import MAX_SOURCE_BYTES, read_file
from scripts.tools.tool_registry import execute_tool


def call(name, **arguments):
    return {'function': {'name': name, 'arguments': json.dumps(arguments)}}


def source_sandbox(content):
    sandbox = Mock()

    def execute(command):
        # Exercise the exact remote paging program with in-memory source text;
        # no Kubernetes, subprocess, filesystem source reads or network access.
        assert command[:2] == ['python3', '-c']
        stdout = io.StringIO()
        with patch('sys.argv', ['-c'] + command[3:]), \
                patch('builtins.open', return_value=io.BytesIO(content.encode('utf-8'))), \
                redirect_stdout(stdout):
            exec(compile(command[2], '<sandbox read_file>', 'exec'), {})
        return SimpleNamespace(stdout=stdout.getvalue().encode(), stderr=b'', exit_code=0)

    sandbox._execute.side_effect = execute
    return sandbox


class BoundedObservationsTest(unittest.TestCase):
    def test_small_output_is_unchanged(self):
        output = json.dumps({'content': 'abc', 'truncated': False})
        self.assertEqual(bounded_tool_output(output), output)

    def test_file_size_cliff_keeps_almost_all_source_and_both_ends(self):
        output = json.dumps({'content': 'START ' + 'x' * 4050 + ' END', 'truncated': False})
        bounded = bounded_tool_output(output)
        parsed = json.loads(bounded)
        self.assertLessEqual(len(bounded), 4000)
        self.assertGreater(len(parsed['content']), 3900)
        self.assertTrue(parsed['content'].startswith('START '))
        self.assertTrue(parsed['content'].endswith(' END'))
        self.assertTrue(parsed['truncated'])

    def test_command_status_and_error_tail_survive_escaping_and_truncation(self):
        output = json.dumps({'stdout': '\n"' * 6000, 'stderr': 'Traceback\n' + 'x' * 9000 + '\nKeyError: account_id',
                             'exit_code': 1, 'timed_out': False, 'error': 'Command failed', 'truncated': False})
        bounded = bounded_tool_output(output)
        parsed = json.loads(bounded)
        self.assertLessEqual(len(bounded), 4000)
        self.assertEqual(parsed['exit_code'], 1)
        self.assertFalse(parsed['timed_out'])
        self.assertEqual(parsed['error'], 'Command failed')
        self.assertIn('Traceback', parsed['stderr'])
        self.assertIn('KeyError: account_id', parsed['stderr'])
        self.assertTrue(parsed['truncated'])

    def test_nested_serialized_json_stays_valid_and_keeps_complete_entries(self):
        records = [{'id': index, 'content': 'decision ' * 70} for index in range(20)]
        output = json.dumps({'status': 'ok', 'context': json.dumps({'memories': records})})
        bounded = bounded_tool_output(output)
        parsed = json.loads(bounded)
        inner = json.loads(parsed['context'])
        self.assertLessEqual(len(bounded), 4000)
        self.assertEqual(parsed['status'], 'ok')
        self.assertTrue(parsed['truncated'])
        self.assertGreater(len(inner['memories']), 0)
        self.assertEqual(inner['memories'], records[:len(inner['memories'])])

    def test_compact_encoding_does_not_claim_source_was_truncated(self):
        output = json.dumps({'content': 'đ' * 1500, 'truncated': False})
        bounded = bounded_tool_output(output)
        self.assertGreater(len(output), 4000)
        self.assertLessEqual(len(bounded), 4000)
        self.assertEqual(json.loads(bounded), json.loads(output))

    def test_large_error_maps_keep_diagnostics_and_status(self):
        output = json.dumps({'status': 'failed', 'exit_code': 2,
                             'errors': {f'file-{index}': 'bad syntax' for index in range(1000)}})
        bounded = bounded_tool_output(output, 512)
        parsed = json.loads(bounded)
        self.assertLessEqual(len(bounded), 512)
        self.assertEqual(parsed['exit_code'], 2)
        self.assertEqual(parsed['status'], 'failed')
        self.assertGreater(len(parsed['errors']), 0)
        self.assertEqual(parsed['errors']['file-0'], 'bad syntax')

    def test_plain_text_and_top_level_lists_fit_valid_json(self):
        for output in ('start ' + 'x' * 7000 + ' end', json.dumps([{'value': 'x' * 500} for _ in range(80)])):
            with self.subTest(output=output[:20]):
                bounded = bounded_tool_output(output, 512)
                self.assertLessEqual(len(bounded), 512)
                self.assertTrue(json.loads(bounded)['truncated'])


class ReadFilePaginationTest(unittest.TestCase):
    def test_legacy_path_only_returns_small_file_unchanged(self):
        source = 'def greet():\n    return "xin chào"\n'
        sandbox = source_sandbox(source)
        result = read_file({'path': 'hello.py'}, sandbox)
        self.assertEqual(result['content'], source)
        self.assertFalse(result['truncated'])
        self.assertEqual(result['start_line'], 1)
        self.assertEqual(result['end_line'], 2)
        self.assertIsNone(result['next_line'])

    def test_explicit_page_has_correct_line_boundaries_and_eof(self):
        sandbox = source_sandbox('one\ntwo\nthree\nfour\nfive')
        result = json.loads(execute_tool(call('read_file', path='text.txt', start_line=2, max_lines=2), sandbox=sandbox))
        self.assertEqual(result['content'], 'two\nthree\n')
        self.assertTrue(result['truncated'])
        self.assertEqual((result['next_line'], result['next_column']), (4, 1))
        last = read_file({'path': 'text.txt', 'start_line': 4, 'max_lines': 2}, sandbox)
        self.assertEqual(last['content'], 'four\nfive')
        self.assertFalse(last['truncated'])
        beyond = read_file({'path': 'text.txt', 'start_line': 99}, sandbox)
        self.assertEqual(beyond['content'], '')
        self.assertFalse(beyond['truncated'])

    def test_pages_reconstruct_large_source_without_losing_lines_or_long_line_columns(self):
        # Escapes, Unicode, CRLF and a single oversized line all need to survive.
        source = '"\\đ"' * 2200 + '\r\n' + ''.join(f'line {number}\n' for number in range(600))
        sandbox = source_sandbox(source)
        arguments = {'path': 'large.py'}
        pages = []
        for _ in range(30):
            output = execute_tool(call('read_file', **arguments), sandbox=sandbox)
            self.assertLessEqual(len(output), 4000)
            self.assertEqual(bounded_tool_output(output), output)
            result = json.loads(output)
            pages.append(result['content'])
            if not result['truncated']:
                break
            arguments.update(start_line=result['next_line'], start_column=result['next_column'])
        else:
            self.fail('Pagination did not reach end of file')
        self.assertGreater(len(pages), 3)
        self.assertEqual(''.join(pages), source)

    def test_path_is_passed_as_an_argument_and_no_shell_is_used(self):
        sandbox = source_sandbox('safe\n')
        path = '$(touch injected); quoted file.py'
        read_file({'path': path}, sandbox)
        command = sandbox._execute.call_args.args[0]
        self.assertEqual(command[:2], ['python3', '-c'])
        self.assertEqual(command[3], '/home/user/project/' + path)
        self.assertNotIn(path, command[2])

    def test_missing_unknown_and_wrong_type_arguments_never_execute(self):
        invalid = [dict(), {'path': 3}, {'path': 'x', 'start_line': True},
                   {'path': 'x', 'start_line': '2'}, {'path': 'x', 'start_line': 1.5},
                   {'path': 'x', 'start_line': 0}, {'path': 'x', 'max_lines': 2001},
                   {'path': 'x', 'start_column': -1}, {'path': 'x', 'unknown': 'a'}]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                sandbox = Mock()
                result = json.loads(execute_tool(call('read_file', **arguments), sandbox=sandbox))
                self.assertIn('invalid arguments', result['error'])
                sandbox._execute.assert_not_called()

    def test_read_failure_preserves_exit_code_and_diagnostic(self):
        sandbox = Mock()
        sandbox._execute.return_value = SimpleNamespace(stdout=b'', stderr=b'FileNotFoundError: missing.py', exit_code=1)
        result = json.loads(execute_tool(call('read_file', path='missing.py'), sandbox=sandbox))
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(result['error'], 'read_file failed')
        self.assertIn('FileNotFoundError', result['stderr'])

    def test_complete_credentials_are_masked_before_pages_split_them(self):
        secret = 'ABC_nonprefixed_synthetic_' * 170
        source = ('password = "' + secret + '"\n'
                  'private_key = """-----BEGIN PRIVATE KEY-----\n' +
                  ('SYNTHETIC_PRIVATE_BODY' * 3 + '\n') * 75 +
                  '-----END PRIVATE KEY-----"""\n'
                  'print("last line")\n')
        sandbox = source_sandbox(source)
        args, pages = {'path': 'config.py'}, []
        for _ in range(10):
            output = execute_tool(call('read_file', **args), sandbox=sandbox)
            page = json.loads(output)
            model_page = json.loads(redact_sensitive_text(output))
            self.assertTrue(page['redacted'])
            self.assertNotIn('ABC_nonprefixed_synthetic_', model_page['content'])
            self.assertNotIn('SYNTHETIC_PRIVATE_BODY', model_page['content'])
            self.assertLessEqual(len(output), 4000)
            pages.append(page['content'])
            if not page['truncated']:
                break
            args.update(start_line=page['next_line'], start_column=page['next_column'])
        else:
            self.fail('Redacted pagination did not reach EOF')
        safe_source = ''.join(pages)
        self.assertEqual(len(safe_source), len(source))
        self.assertEqual([i for i, char in enumerate(source) if char == '\n'],
                         [i for i, char in enumerate(safe_source) if char == '\n'])
        self.assertTrue(safe_source.endswith('print("last line")\n'))

    def test_single_source_assignment_survives_repeated_model_filtering(self):
        source = 'password = provided_password\n'
        output = execute_tool(call('read_file', path='auth.py'), sandbox=source_sandbox(source))
        self.assertEqual(json.loads(redact_sensitive_text(output))['content'], source)
        self.assertFalse(json.loads(output)['redacted'])

    def test_large_source_is_rejected_before_full_scan(self):
        output = execute_tool(call('read_file', path='huge.py'),
                              sandbox=source_sandbox('x' * (MAX_SOURCE_BYTES + 1)))
        self.assertIn('2 MiB file limit', json.loads(output)['error'])

    def test_nested_json_credentials_cannot_escape_through_later_pages(self):
        source = json.dumps({'password': {'type': 'string', 'value': 'synthetic_nested_' * 700},
                             'safe': 'keep this field'})
        sandbox = source_sandbox(source)
        args, pages = {'path': 'settings.json'}, []
        for _ in range(10):
            page = read_file(args, sandbox)
            self.assertNotIn('synthetic_nested_', page['content'])
            pages.append(page['content'])
            if not page['truncated']:
                break
            args.update(start_line=page['next_line'], start_column=page['next_column'])
        else:
            self.fail('JSON pagination did not reach EOF')
        safe = ''.join(pages)
        self.assertEqual(len(safe), len(source))
        self.assertEqual(json.loads(safe)['safe'], 'keep this field')


if __name__ == '__main__':
    unittest.main()

"""Regression tests for source fidelity and shared credential detection."""
import ast
import json
import unittest

from memory.privacy import (contains_sensitive_value, redact_sensitive_text,
                            redact_source_text, redact_file_layout, sanitize_json_value)


class MemoryPrivacyTests(unittest.TestCase):
    def assert_safe(self, value):
        self.assertFalse(contains_sensitive_value(value))
        if isinstance(value, str):
            self.assertEqual(redact_sensitive_text(value), value)
        else:
            self.assertEqual(sanitize_json_value(value), value)

    def test_source_annotations_and_expressions_are_unchanged(self):
        sources = [
            'password: str',
            'password: "str"',
            'password: Optional[str]',
            'password = input("Password: ")',
            'api_key = os.getenv("OPENAI_API_KEY")',
            'api_key = os.environ["OPENAI_API_KEY"]',
            'password = credentials.password',
            'password = process.env.PASSWORD',
            'password = None',
            'const password: string = process.env.PASSWORD;',
            'def login(password: str) -> bool:\n    return bool(password)',
            'def login(password: "SecretValue"):\n    return password',
            'def login(password: str):',
            '# mật khẩu\npassword: "SecretValue"\npassword = input("Password: ")',
            'password = f"{os.getenv(\'PASSWORD\')}"',
            'PASSWORD=${DATABASE_PASSWORD}',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assert_safe(source)

    def test_known_credentials_are_removed_and_detection_agrees(self):
        sources = [
            'api_key=short', 'password=abc', 'refresh_token=small',
            'password="tiny"', "password='spaces in secret'",
            'Authorization: abcdef', 'Bearer short-token',
            'Authorization: Basic c2VjcmV0',
            'github_pat_1234567890abcdefgh', 'AKIA1234567890ABCDEF',
            'sk-or-v1-' + 'x' * 40,
            'postgresql://user:p4ssword@db.example.test/database',
            '-----BEGIN RSA PRIVATE KEY-----\nSECRET BODY\n-----END RSA PRIVATE KEY-----',
            '-----BEGIN PRIVATE KEY-----\nINCOMPLETE SECRET BODY',
            'OPENAI_API_KEY=short', 'DATABASE_PASSWORD=short',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(contains_sensitive_value(source))
                clean = redact_sensitive_text(source)
                self.assertIn('[REDACTED]', clean)
                self.assert_safe(clean)

    def test_literal_redaction_keeps_python_syntax(self):
        for source in [
            'password = "secret-value"',
            'password: str = "secret-value"',
            'password: "SecretValue" = "secret-value"',
            'password = r"secret-value"',
            'password = b"secret-value"',
            'password = f"secret-value"',
            'def login(password: str = "secret-value"):\n    return password',
            'password = "secret\\\"value"',
            "password = '''multi\nline secret'''",
            'os.environ["PASSWORD"] = "secret-value"',
        ]:
            with self.subTest(source=source):
                clean = redact_sensitive_text(source)
                self.assertNotEqual(clean, source)
                ast.parse(clean)
                self.assertIn('[REDACTED]', clean)
                self.assert_safe(clean)

    def test_json_code_and_schema_are_preserved(self):
        value = {
            'content': 'def authenticate(password: str):\n    return password',
            'parameters': {
                'type': 'object',
                'properties': {'password': {'type': 'string', 'description': 'A user password'}},
            },
        }
        self.assert_safe(value)
        source = json.dumps(value, indent=2)
        self.assert_safe(source)

    def test_nested_credentials_and_schema_defaults_are_redacted(self):
        value = {
            'nested': [{'private_key': 'shortprivate', 'password': 'shortpassword'}],
            'schema': {'properties': {'password': {
                'type': 'string', 'description': 'A user password',
                'default': 'shortdefault', 'examples': ['shortexample'],
            }}},
            'content': 'password = "dummy_secret"',
        }
        self.assertTrue(contains_sensitive_value(value))
        clean = sanitize_json_value(value)
        for secret in ('shortprivate', 'shortpassword', 'shortdefault', 'shortexample', 'dummy_secret'):
            self.assertNotIn(secret, json.dumps(clean))
        schema = clean['schema']['properties']['password']
        self.assertEqual(schema['type'], 'string')
        self.assertEqual(schema['description'], 'A user password')
        self.assert_safe(clean)
        self.assertEqual(json.loads(redact_sensitive_text(json.dumps(value))), clean)

    def test_json_encoded_source_does_not_hide_credentials(self):
        raw = json.dumps({'function': {'arguments': json.dumps({'content': 'password = "shortsecret"'})}})
        self.assertTrue(contains_sensitive_value(raw))
        clean = redact_sensitive_text(raw)
        self.assertNotIn('shortsecret', clean)
        nested = json.loads(json.loads(clean)['function']['arguments'])
        ast.parse(nested['content'])
        self.assert_safe(clean)

    def test_authorization_literal_keeps_its_quotes(self):
        self.assertEqual(redact_sensitive_text('Authorization: "opaque secret"'),
                         'Authorization: "[REDACTED]"')

    def test_prompt_label_cannot_consume_following_source(self):
        for separator in ('\n', '; '):
            source = ('password = input("Enter password: ")' + separator +
                      'secret = os.environ["APP_SECRET"]' + separator +
                      'api_key = os.getenv("API_KEY")')
            self.assert_safe(source)
            combined = source + '\nsaved_password = "synthetic-secret"\n'
            clean = redact_sensitive_text(combined)
            self.assertIn(source, clean)
            self.assertNotIn('synthetic-secret', clean)
            ast.parse(clean)
        source = 'password = input("Enter password: "); secret = "actual-secret"'
        clean = redact_sensitive_text(source)
        self.assertIn('input("Enter password: ")', clean)
        self.assertNotIn('actual-secret', clean)
        ast.parse(clean)

    def test_sensitive_nested_object_cannot_hide_plain_values(self):
        value = {'password': {'value': 'nestedsecret'}}
        self.assertTrue(contains_sensitive_value(value))
        self.assertEqual(sanitize_json_value(value), {'password': {'value': '[REDACTED]'}})

    def test_environment_placeholders_remain_references(self):
        for value in ({'password': '${PASSWORD}'}, {'api_key': '${OPENAI_API_KEY:-}'},
                      {'password': '$PASSWORD'}, {'api_key': '{{ secrets.OPENAI_API_KEY }}'}):
            self.assert_safe(value)

    def test_parameter_references_and_attribute_assignments_preserve_source(self):
        source = ('def login(password, supplied_password):\n'
                  '    self.password = password\n'
                  '    password = supplied_password\n'
                  '    return password\n')
        self.assert_safe(source)
        # The file path disambiguates Python names from bare .env values, even
        # on an isolated page or in serialized write_file arguments.
        for source in ('password = supplied_password\n', '    password = supplied_password\n'):
            self.assert_safe({'path': 'auth.py', 'content': source})
        self.assertIn('[REDACTED]', redact_sensitive_text('password=supplied_password'))

    def test_partial_source_keeps_labels_and_forward_annotations(self):
        source = '    password = input("Enter password: "); secret = "real_secret"\n'
        clean = redact_sensitive_text(source)
        self.assertEqual(clean, '    password = input("Enter password: "); secret = "[REDACTED]"\n')
        self.assert_safe('def login(\n    password: "SecretValue",\n')
        self.assert_safe({'path': 'types.py', 'content': '    password: "SecretValue"\n'})

    def test_typed_credential_objects_are_not_mistaken_for_schemas(self):
        for kind in ('plaintext', 'string'):
            value = {'password': {'type': kind, 'value': 'synthetic-secret'}}
            self.assertTrue(contains_sensitive_value(value))
            self.assertNotIn('synthetic-secret', json.dumps(sanitize_json_value(value)))

    def test_layout_masking_preserves_original_character_and_line_offsets(self):
        source = ('password = "x"\r\n'
                  'private_key = """-----BEGIN PRIVATE KEY-----\n'
                  'SYNTHETIC KEY BODY\n-----END PRIVATE KEY-----"""\n'
                  'print("xin chào")\n')
        clean = redact_source_text(source, preserve_layout=True)
        self.assertEqual(len(source), len(clean))
        self.assertEqual([i for i, char in enumerate(source) if char in '\r\n'],
                         [i for i, char in enumerate(clean) if char in '\r\n'])
        self.assertNotIn('SYNTHETIC KEY BODY', clean)
        self.assertNotIn('password = "x"', clean)
        self.assertTrue(clean.endswith('print("xin chào")\n'))
        ast.parse(clean)

    def test_file_paths_distinguish_yaml_values_from_python_annotations(self):
        for source in ('password: "SecretValue"\n', 'password: SecretValue\n'):
            self.assert_safe({'path': 'types.py', 'content': source})
            configuration = {'path': 'settings.yaml', 'content': source}
            self.assertTrue(contains_sensitive_value(configuration))
            self.assertNotIn('SecretValue', sanitize_json_value(configuration)['content'])
        self.assertEqual(redact_sensitive_text('password: "actual_secret"'), 'password: "[REDACTED]"')

    def test_nested_json_layout_masks_credentials_without_changing_offsets(self):
        value = {'credentials': {'password': {'type': 'string', 'value': 'x' * 9000}},
                 'safe': [{'answer': 42}], 'secret': [123, True, 'has\\escape"chars', {}, []]}
        source = json.dumps(value, indent=2)
        clean = redact_file_layout(source, 'settings.json')
        self.assertEqual(len(clean), len(source))
        self.assertEqual([i for i, char in enumerate(clean) if char == '\n'],
                         [i for i, char in enumerate(source) if char == '\n'])
        parsed = json.loads(clean)
        self.assertEqual(parsed['safe'], value['safe'])
        self.assertNotEqual(parsed['secret'], value['secret'])
        self.assertNotIn('x' * 30, clean)


if __name__ == '__main__':
    unittest.main()

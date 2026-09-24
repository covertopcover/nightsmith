"""Tests for serve.py's request checks. Stdlib only, no mlx:

    python3 -m unittest serve_test
"""

import os
import tempfile
import unittest

from serve import DEFAULT, check_request, error_body, health_state, model_aliases

REPO = "mlx-community/gemma-4-12B-it-4bit"
PATH = "/Users/x/.nightsmith/models/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/73bc"
ALIASES = model_aliases(REPO, PATH)
CHAT = "/v1/chat/completions"
OK_MESSAGES = [{"role": "user", "content": "Say OK"}]


def check(body, path=CHAT):
    return check_request(path, body, ALIASES, REPO)


class Messages(unittest.TestCase):
    def test_missing_messages_is_a_400_naming_the_field(self):
        # The stock server closed the connection with no response at all.
        err = check({"model": DEFAULT, "max_tokens": 3})
        self.assertEqual(err.status, 400)
        self.assertEqual(err.param, "messages")

    def test_empty_messages_is_a_400_not_a_404(self):
        err = check({"messages": []})
        self.assertEqual((err.status, err.param), (400, "messages"))

    def test_messages_that_are_not_a_list(self):
        self.assertEqual(check({"messages": "hi"}).status, 400)

    def test_a_message_without_a_role(self):
        err = check({"messages": [{"content": "hi"}]})
        self.assertEqual(err.status, 400)
        self.assertIn("messages[0]", err.message)

    def test_both_chat_paths_are_checked(self):
        self.assertEqual(check({}, path="/chat/completions").status, 400)

    def test_text_completion_needs_a_prompt(self):
        err = check({}, path="/v1/completions")
        self.assertEqual((err.status, err.param), (400, "prompt"))
        self.assertIsNone(check({"prompt": "hi"}, path="/v1/completions"))


class Model(unittest.TestCase):
    def test_every_name_for_the_served_model_is_accepted(self):
        for model in (DEFAULT, REPO, PATH):
            with self.subTest(model=model):
                self.assertIsNone(check({"model": model, "messages": OK_MESSAGES}))

    def test_leaving_model_out_is_accepted(self):
        self.assertIsNone(check({"messages": OK_MESSAGES}))

    def test_the_resolved_path_is_accepted_too(self):
        with tempfile.TemporaryDirectory() as d:
            real = os.path.join(d, "real")
            os.mkdir(real)
            link = os.path.join(d, "link")
            os.symlink(real, link)
            aliases = model_aliases(REPO, link)
            body = {"model": os.path.realpath(real), "messages": OK_MESSAGES}
            self.assertIsNone(check_request(CHAT, body, aliases, REPO))

    def test_unknown_model_is_a_404_that_names_the_working_id(self):
        # Must be refused here: the stock server unloads the served model
        # before it discovers the other one does not exist.
        err = check({"model": "not-a-model", "messages": OK_MESSAGES})
        self.assertEqual((err.status, err.code), (404, "model_not_found"))
        self.assertIn(REPO, err.message)

    def test_empty_model_is_a_400_without_internals(self):
        err = check({"model": "", "messages": OK_MESSAGES})
        self.assertEqual(err.status, 400)
        self.assertNotIn("config.json", err.message)

    def test_non_string_model(self):
        self.assertEqual(check({"model": 3, "messages": OK_MESSAGES}).status, 400)

    def test_draft_model_and_adapters_are_refused(self):
        # Both reach the same unload-then-load path as an unknown model.
        self.assertEqual(check({"messages": OK_MESSAGES, "draft_model": "x/y"}).param,
                         "draft_model")
        self.assertEqual(check({"messages": OK_MESSAGES, "adapters": "a"}).param,
                         "adapters")
        self.assertIsNone(check({"messages": OK_MESSAGES, "draft_model": DEFAULT,
                                 "adapters": None}))


class ResponseFormat(unittest.TestCase):
    # Nothing constrains the output, so a request for JSON or a schema would
    # get a 200 and whatever the model wrote. Refuse it instead.
    def test_json_schema_is_a_400_naming_the_field(self):
        err = check({"messages": OK_MESSAGES, "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "c", "schema": {"type": "object"}}}})
        self.assertEqual((err.status, err.param), (400, "response_format"))

    def test_json_object_is_a_400(self):
        err = check({"messages": OK_MESSAGES,
                     "response_format": {"type": "json_object"}})
        self.assertEqual((err.status, err.param), (400, "response_format"))

    def test_text_and_none_are_what_it_does_anyway(self):
        self.assertIsNone(check({"messages": OK_MESSAGES,
                                 "response_format": {"type": "text"}}))
        self.assertIsNone(check({"messages": OK_MESSAGES, "response_format": None}))

    def test_not_an_object(self):
        err = check({"messages": OK_MESSAGES, "response_format": "json"})
        self.assertEqual(err.status, 400)

    def test_text_completions_too(self):
        err = check({"prompt": "hi", "response_format": {"type": "json_object"}},
                    path="/v1/completions")
        self.assertEqual(err.param, "response_format")


class StreamOptions(unittest.TestCase):
    """The pinned server indexes include_usage directly, so a stream_options
    without it raises after the headers are sent — which reaches a client as a
    stream that stops for no stated reason. Refused before a byte is written.
    """

    def test_bare_stream_options_is_a_400(self):
        err = check({"messages": OK_MESSAGES, "stream": True, "stream_options": {}})
        self.assertEqual((err.status, err.param), (400, "stream_options"))

    def test_include_usage_must_be_a_bool(self):
        err = check({"messages": OK_MESSAGES, "stream_options": {"include_usage": "yes"}})
        self.assertEqual((err.status, err.param), (400, "stream_options"))

    def test_not_an_object(self):
        err = check({"messages": OK_MESSAGES, "stream_options": True})
        self.assertEqual((err.status, err.param), (400, "stream_options"))

    def test_what_nightsmith_sends_is_accepted(self):
        self.assertIsNone(check({"messages": OK_MESSAGES, "stream": True,
                                 "stream_options": {"include_usage": True}}))

    def test_absent_and_null_are_fine(self):
        self.assertIsNone(check({"messages": OK_MESSAGES, "stream": True}))
        self.assertIsNone(check({"messages": OK_MESSAGES, "stream_options": None}))


class Health(unittest.TestCase):
    def test_states(self):
        self.assertEqual(health_state(True, False), "loading")
        self.assertEqual(health_state(True, True), "ready")
        self.assertEqual(health_state(False, True), "failed")
        self.assertEqual(health_state(False, False), "failed")

    def test_stopping_wins(self):
        self.assertEqual(health_state(True, True, stopping=True), "stopping")
        self.assertEqual(health_state(False, True, stopping=True), "stopping")


class ErrorBody(unittest.TestCase):
    def test_openai_shape(self):
        body = error_body(check({}))
        self.assertEqual(set(body["error"]), {"message", "type", "param", "code"})
        self.assertEqual(body["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()

"""Tests for serve.py's request checks. Stdlib only, no mlx:

    python3 -m unittest serve_test
"""

import os
import tempfile
import unittest

from serve import (
    CONCURRENT_PROMPT_TOKENS,
    DEFAULT,
    MAX_CONCURRENT,
    MAX_PROMPT_TOKENS,
    ClientError,
    admit,
    check_request,
    check_size,
    measure_prompt,
    error_body,
    estimate_tokens,
    health_state,
    model_aliases,
    parse_body,
    prompt_text,
    split_args,
)

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


class Roles(unittest.TestCase):
    """A role the template never heard of used to be folded into the prompt
    and answered with a 200, so a misspelling looked like it worked."""

    def test_the_openai_roles_are_accepted(self):
        for role in ("system", "user", "assistant", "tool"):
            with self.subTest(role=role):
                self.assertIsNone(check({"messages": [{"role": role, "content": "x"}]}))

    def test_a_misspelled_role_is_a_400_naming_the_message(self):
        err = check({"messages": [{"role": "sytem", "content": "x"}]})
        self.assertEqual((err.status, err.param), (400, "messages"))
        self.assertIn("messages[0]", err.message)
        self.assertIn("sytem", err.message)


class Body(unittest.TestCase):
    """The two replies that were not in the OpenAI error shape. A client
    reading err["error"]["message"] threw on exactly these paths."""

    def test_valid_json_object(self):
        self.assertEqual(parse_body(b'{"a": 1}'), {"a": 1})

    def test_malformed_json_is_a_400_in_the_error_shape(self):
        with self.assertRaises(ClientError) as cm:
            parse_body(b'{"messages": [')
        self.assertEqual(cm.exception.status, 400)
        self.assertEqual(set(error_body(cm.exception)["error"]),
                         {"message", "type", "param", "code"})

    def test_a_json_array_is_not_a_request(self):
        with self.assertRaises(ClientError) as cm:
            parse_body(b'[1, 2]')
        self.assertEqual(cm.exception.status, 400)

    def test_undecodable_bytes(self):
        with self.assertRaises(ClientError) as cm:
            parse_body(b'\xff\xfe')
        self.assertEqual(cm.exception.status, 400)


class PromptSize(unittest.TestCase):
    """Measured on a 16 GB M4: 43,389 tokens answered in 406 s, ~58,000 killed
    the server with no reply at all."""

    def big(self, tokens):
        return [{"role": "user", "content": "x" * (tokens * 4)}]

    def sized(self, body, path=CHAT, count=None):
        size, counted = measure_prompt(path, body, count)
        return size, check_size(size, counted, path)

    def test_an_ordinary_prompt_passes(self):
        size, err = self.sized({"messages": OK_MESSAGES})
        self.assertIsNone(err)
        self.assertLess(size, 10)

    def test_past_the_ceiling_is_a_400_before_the_model_sees_it(self):
        size, err = self.sized({"messages": self.big(MAX_PROMPT_TOKENS + 100)})
        self.assertEqual((err.status, err.code), (400, "prompt_too_large"))
        self.assertIn("58,000", err.message)  # it says what it is protecting
        self.assertGreater(size, MAX_PROMPT_TOKENS)

    def test_the_ceiling_is_under_what_was_measured_to_work(self):
        self.assertLess(MAX_PROMPT_TOKENS, 43_389)

    def test_a_whole_conversation_counts_not_just_the_last_turn(self):
        half = MAX_PROMPT_TOKENS * 2 // 3
        body = {"messages": [{"role": "user", "content": "x" * (half * 4)},
                             {"role": "assistant", "content": "y" * (half * 4)}]}
        self.assertIsNotNone(self.sized(body)[1])

    def test_text_completions_are_measured_too(self):
        body = {"prompt": "x" * ((MAX_PROMPT_TOKENS + 100) * 4)}
        self.assertEqual(self.sized(body, path="/v1/completions")[1].param, "prompt")

    def test_the_model_counts_when_it_can_and_the_message_stops_hedging(self):
        # Characters over four is wrong in both directions — it over-counts
        # repetition and under-counts code and anything not mostly ASCII.
        # Under-counting is how a fatal prompt gets past a check meant to stop
        # it, so the real tokenizer is used whenever there is one.
        body = {"messages": [{"role": "user", "content": "x" * 400}]}
        self.assertEqual(measure_prompt(CHAT, body), (100, False))
        self.assertEqual(measure_prompt(CHAT, body, lambda t: 900), (900, True))

        counted = check_size(MAX_PROMPT_TOKENS + 1, True, CHAT)
        self.assertNotIn("about", counted.message)
        estimated = check_size(MAX_PROMPT_TOKENS + 1, False, CHAT)
        self.assertIn("about", estimated.message)

    def test_a_tokenizer_that_throws_falls_back_rather_than_500ing(self):
        def broken(_):
            raise RuntimeError("no")
        body = {"messages": [{"role": "user", "content": "x" * 400}]}
        self.assertEqual(measure_prompt(CHAT, body, broken), (100, False))

    def test_content_parts_are_read(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x" * 400}, {"type": "text", "text": "y" * 400}]}]}
        self.assertEqual(estimate_tokens(prompt_text(CHAT, body)), 201)

    def test_a_malformed_body_does_not_throw_on_the_way_to_its_400(self):
        # check_request refuses these; measuring must not raise first.
        for body in ({"messages": "hi"}, {"messages": [1, 2]}, {}):
            with self.subTest(body=body):
                self.assertIsNone(self.sized(body)[1])


class Admission(unittest.TestCase):
    """The ladder, exactly as it was measured: 3 x 8,000 fine, 4 x 4,000 fatal
    (twice), 5 x 2,000 fatal. Parallel streams dominate, not combined size."""

    def test_one_request_alone_always_runs(self):
        self.assertTrue(admit(MAX_PROMPT_TOKENS, 0, 0))

    def test_three_at_eight_thousand_is_admitted(self):
        self.assertTrue(admit(8_000, 1, 8_000))
        self.assertTrue(admit(8_000, 2, 16_000))

    def test_a_fourth_waits_however_small(self):
        self.assertFalse(admit(10, MAX_CONCURRENT, 300))

    def test_five_small_ones_never_run_together(self):
        self.assertTrue(admit(2_000, 1, 2_000))
        self.assertTrue(admit(2_000, 2, 4_000))
        self.assertFalse(admit(2_000, 3, 6_000))

    def test_the_combined_budget_holds_below_the_count(self):
        self.assertFalse(admit(20_000, 1, 8_000))
        self.assertLessEqual(CONCURRENT_PROMPT_TOKENS, 24_000)

    def test_a_large_prompt_waits_until_it_is_alone_rather_than_deadlocking(self):
        big = CONCURRENT_PROMPT_TOKENS + 1_000
        self.assertFalse(admit(big, 1, 100))
        self.assertTrue(admit(big, 0, 0))


class Args(unittest.TestCase):
    def test_nightsmith_flags_are_taken_out_of_the_mlx_command_line(self):
        repo, drain, rest = split_args(
            ["--served-model-id", "a/b", "--drain-seconds", "150",
             "--model", "/p", "--port", "8080"])
        self.assertEqual((repo, drain), ("a/b", 150))
        self.assertEqual(rest, ["--model", "/p", "--port", "8080"])

    def test_drain_seconds_is_optional(self):
        repo, drain, rest = split_args(["--served-model-id", "a/b", "--port", "1"])
        self.assertEqual((repo, drain, rest), ("a/b", None, ["--port", "1"]))

    def test_a_missing_model_id_is_refused(self):
        with self.assertRaises(ValueError):
            split_args(["--port", "8080"])

    def test_a_drain_that_is_not_a_number_is_refused(self):
        with self.assertRaises(ValueError):
            split_args(["--served-model-id", "a/b", "--drain-seconds", "soon"])


if __name__ == "__main__":
    unittest.main()
